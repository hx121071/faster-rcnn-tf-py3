from faster_rcnn.config import cfg
import roi_data_layer_py3.roidb as rdl_roidb
from roi_data_layer_py3.layer import RoIDataLayer
from utils_py3.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time

class SolverWrapper(object):
    """
    A simple wrapper for train faster_rcnn
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir,
                 pretrained_model=None, max_iters=40000):
        """
        Initialize the SolverWrapper
        """
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        #### 我认为下面的代码你影响训练过程
        # print 'Computing bounding-box regression targets...'
        # if cfg.TRAIN.BBOX_REG:
        #     self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        # print 'done'
        self.saver = saver
        self.sess = sess
        self.max_iters = max_iters
    def snapshot(self, iter):
        net = self.net
        ########################################################下面这写代码没搞定##############################################
        # if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
        #     # save original values
        #     with tf.variable_scope('bbox_pred', reuse=True):
        #         weights = tf.get_variable("weights")
        #         biases = tf.get_variable("biases")
        #
        #     orig_0 = weights.eval()
        #     orig_1 = biases.eval()
        #
        #     # scale and shift with bbox reg unnormalization; then save snapshot
        #     weights_shape = weights.get_shape().as_list()
        #     self.sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
        #     self.sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')

        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')

        filename = os.path.join(self.output_dir, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        ##########################################同样的问题####################################################
        # if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
        #     with tf.variable_scope('bbox_pred', reuse=True):
        #         # restore net to original state
        #         sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
        #         sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2, if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2, otherwise
        """
        sigma2 = sigma * sigma
        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                            tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)#这个是除以batch_size
        return outside_mul
    def train_model(self):
        """
        Network training loop
        """

        ## 获得data
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)


        ## 需要获取的值并计算损失，分为RPN 和 fast-rcnn

        ## RPN

        ##　classification loss

        ##  anchors predicted
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'), \
                                [-1,2])
        ## anchors targets
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0], [-1])

        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])

        rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])

        ### cross_entropy loss

        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        ## rpn bbox_reg L1 loss

        ## anchors predicted
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')

        ## anchors targets
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1], [0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2], [0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3], [0,2,3,1])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3,]))

        ## fast-rcnn

        ## classification loss
        ## bbox predicted
        ## bbox targets
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi-data')[1], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels =label))

        ## bbox_reg L1 loss

        ## bbox_res predicted
        bbox_pred = self.net.get_output('bbox_pred')

        ## bbox_reg targets
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]

        smoothl1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        loss_box = tf.reduce_mean(tf.reduce_sum(smoothl1, reduction_indices=[1]))


        ## total loss

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        ## learning rate,optimizer

        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)

        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)

        ## initializer variables
        self.sess.run(tf.global_variables_initializer())

        ## load pretrained_model
        if self.pretrained_model is not None:
            print('Loading pretrained model '
                    'weights from {:s}'.format(self.pretrained_model))
            self.net.load(self.pretrained_model, self.sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        ## iters

        for iter in range(self.max_iters):
            # get one batch
            blobs = data_layer.forward()
            feed_dict = {self.net.data: blobs['data'],
                         self.net.im_info: blobs['im_info'],
                         self.net.keep_prob: 0.5,
                         self.net.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                run_metadata = rf.RunMetadata()

            timer.tic()

            rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, _ = \
                                self.sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
                                                                                                feed_dict=feed_dict,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)
            timer.toc()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print ('iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, self.max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval()))
                print ('speed: {:.3f}s / iter'.format(timer.average_time))

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(iter)

        if last_snapshot_iter != iter:
            self.snapshot(iter)

        ## snap_shot
def get_training_roidb(imdb):
    if cfg.TRAIN.USE_FLEPPED:
        print('Appending horizontally-flipped train example...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')

    rdl_roidb.prepare_roidb(imdb)

    return imdb.roidb

def get_data_layer(roidb, num_classes):
    """
    return a data layer
    """

    layer = RoIDataLayer(roidb, num_classes)

    return layer
def train_net(network, imdb, roidb, output_dir,
             pretrained_model=None, max_iters=40000):
    """
    Train a Fast R-CNN network.
    """
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir,
                            pretrained_model = pretrained_model, max_iters=max_iters)
        print('Solving...')
        sw.train_model()
        print('done solving')
