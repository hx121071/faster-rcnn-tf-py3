import numpy as np
import yaml
from faster_rcnn.config import cfg
from rpn.generate_anchors import generate_anchors
from faster_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.nms_wrapper import nms
import pdb

DEBUG = False
"""
Ouputs object detection proposals by applying estimated bounding-box
tranformations to a set of regular boxes (called "anchors")
"""

def proposal_layer(rpn_cls_prob,rpn_bbox_pred,im_info,\
                    cfg_key,_feat_stride=[16,],anchor_scales=[8, 16, 32]):
    # Algorithm:
    #
    # for each (H, W) location i:
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (ROI top, scores top)
    # layer_params = yaml.load(self.param_str)
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    rpn_cls_prob_reshape = np.transpose(rpn_cls_prob, [0,3,1,2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0,3,1,2])

    im_info = im_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1,\
        'Only single item batches are supported'

    # cfg_key = 'TRAIN' or 'TEST'
    cfg_key = cfg_key.decode('ascii')#这里要添加decode('ascii')

    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    min_size = cfg[cfg_key].RPN_MIN_SIZE


    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred

    if DEBUG:
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))

    # 1. Generate proposals from bbox deltas  and shifted anchors

    height, width = scores.shape[-2:]

    if DEBUG:
        print('score map size: {}'.format(scores.shape))

    #Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # print(shift_x.shape)
    # print(shift_y.shape)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchor:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to  (K*A, 4) shifted anchors

    A = _num_anchors
    K = shifts.shape[0]

    anchors = _anchors.reshape(1, A, 4)+\
                shifts.reshape(1, K, 4).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order

    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same story for the scores:
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)

    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals  via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    print(min_size, im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])
    print("keep is:",keep)
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pro_nms _ropN
    order = scores.ravel().argsort()[::-1]

    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]

    proposals = proposals[order, :]
    scores = scores[order]

    # 6 apply nms (e.g. threshpld = 0.7)
    # 7 take after_nms_topN (e.g. 300)
    # print(type(proposals[0][1]))
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    # print(keep)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    # Output rois blob
    # Our RPN implementation only supports a single input image,
    # so all batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1),dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob

def _filter_boxes(boxes, min_size):
    return np.where(((boxes[:,2]-boxes[:,0] + 1) >= min_size) &
                    ((boxes[:,3]-boxes[:,1] + 1) >= min_size))[0]


if __name__ == '__main__':
    im_info = [[1280 , 720 , 1.0]]
    # gt_boxes = np.array([[553, 331, 783, 611, 1]])
    rpn_cls_prob = np.random.randn(1, int(1280/16), int(720/16), 18).astype(np.float32)
    rpn_bbox_pred = np.random.randn(1, int(1280/16), int(720/16), 36).astype(np.float32)
    cfg_key = 'TRAIN'
    blob = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key)

    print(blob)
