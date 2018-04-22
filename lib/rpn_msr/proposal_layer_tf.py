import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv,clip_boxes
from fast_rcnn.nms_wrapper import nms
import pdb

DEBUG =False
"""
Output object detection proposals by applying estimated bounding-box
transformatoins to a set of regular boxes(called "anchors")
"""

def proposal_layer(rpn_cls_prob_reshape,rpn_bbox_pred,im_info,cfg_key,_feat_stride=[16,],\
anchors_scales=[8,16,32]):
    #Algorithom
    #
    #for each(H,W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox at cell i to each of the A anchors
    #clip_predicted boxes to image
    #remove predicted boxes with either height or width < threshold
    #sort all (proposal,scores) pairs by score from highest to lowest
    #take top pre_nms_topN proposal before nms
    #apply NMS with threshold 0.7 to remaining proposals
    #take after_nms_topN proposals after nms
    #return the top proposals (->ROI top,scores top)
    #layer_params=yaml.load(self.param_str_)

    _anchors =generate_anchors(scales=np.array(anchors_scales))
    _num_anchors=_anchors.shape[0]
    rpn_cls_prob_reshape=np.transpose(rpn_cls_prob_reshape,[0,3,1,2])
    rpn_bbox_pred=np.transpose(rpn_bbox_pred,[0,3,1,2])

    im_info=im_info[0]

    assert rpn_cls_prob_reshape==1,\
            'Only single item batches are supported'

    pre_nms_topN=cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN=cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh=cfg[cfg_key].RPN_NMS_THRESH
    min_size=cfg[cfg_key].RPN_MIN_SIZE


    scores=rpn_cls_prob_reshape[:,_num_anchors:,:,:]
    bbox_deltas=rpn_bbox_pred
    height,width= scores[-2:]

    shift_x=np.arange(0,width)*_feat_stride
    shift_y=np.arange(0,height)*_feat_stride
    shift_x,shift_y=np.meshgrid(shift_x,shift_y)
    shifts=np.vstack((shift_x.ravel(),shift_y.ravel(),
                      shift_x.ravel(),shift_y.ravel())).transpose()#4*height*width x 4

    #Enumerate all shifted anchors:
    #
    #add A anchors (1,A,4) to cell K shifts (K,1,4)  to get shfit anchors (K,A,4)
    #reshape to (K*A,4) shifted anchors
    A=_num_anchors
    K=shifts.shape[0]
    ##why
    anchors=_anchors.reshape((1,A,4))+\
            shifts.reshape(1,K,4).transpose((1,0,2))


    # # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas=bbox_deltas.transpose((0,2,3,1)).reshape((-1,4))

     # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores=scores.tranpose((0,2,3,1)).reshape(-1,1)

    #Convert anchors into proposals via bbox transformatoins

    proposals=bbox_transform_inv(anchors,bbox_deltas)

    #2 clip predicted boxes to image
    proposals =clip_boxes(proposals,images[:2])

    #3 remove predicted boxes with either height or width <threshold
    keep=_filter_boxes(proposals,min_size*im_info[2])
    proposals=proposals[keep,:]
    scores=scores[keep]

    #4 sort all (proposals,scores)
    order=scores.ravel().argsort()[::-1]
    if pre_nms_topN>0:
        order=order[:pre_nms_topN]
    #5 take top pre_nms_topN
    proposals=proposals[order,:]
    scores=scores[order]

    #6 apply nms(eg threshold=0.7)
    #7 take after_nms_topN(eg 300)
    #8 return the top proposals(->ROIs top)

    keep=nms(np.hstack((proposals,scores)),nms_thresh)
    if post_nms_topN>0:
        keep=keep[:post_nms_topN]

    proposals=proposals[keep,:]
    scores=scores[keep]

    #output rois blob
    #Our RPN implementation only supports a single input image ,so all batches inds are 0
    batch_inds=np.zeros((proposals.shapep[0],1),dtype=np.float32)
    blob=np.hstack((batch_inds,proposals.astype(np.float32,copy=False)))
    return blob




def _filter_boxes(proposals,min_size):
    """Remove all boxes with any side smaller than min size"""

    ws=proposals[:,2]-proposals[:,0]+1
    hs=proposals[:,3]-proposals[:,1]+1

    keep=np.where((ws>=min_size) & (hs>=min_size))[0]

    return keep
