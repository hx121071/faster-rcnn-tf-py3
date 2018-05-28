from network.VGGnet_train import VGGnet_train
import tensorflow as tf
import numpy as np
import cv2

if  __name__ == '__main__':
    net = VGGnet_train()
    import numpy as np
    data = cv2.imread("picture0.jpg")
    data = data.reshape((1,)+data.shape).astype(np.float32)
    im_info = [[1280, 720, 1.0]]
    gt_boxes = np.array([[553, 331, 783, 611.0, 1]])
    cls_pro = net.get_output('cls_prob')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, 'VGGnet_fast_rcnn_iter_45000.ckpt')
        cls_p = sess.run(cls_pro, feed_dict = {net.data : data, \
                                                net.im_info : im_info,
                                                net.gt_boxes : gt_boxes})
        print(cls_p.shape)
