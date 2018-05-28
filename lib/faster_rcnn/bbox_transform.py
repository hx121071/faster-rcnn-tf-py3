import numpy as np

def  bbox_transform(ex_rois, gt_rois):
    """
    根据sample roi对照gt rois计算相应的回归函数
    """
    ex_width = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_height = ex_rois[:, 3] - ex_rois[:, 1] + 1.0

    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_width
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_height

    gt_width = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_height = gt_rois[:, 3] - gt_rois[:, 1] + 1.0

    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_height

    target_x = (gt_ctr_x - ex_ctr_x) / ex_width
    target_y = (gt_ctr_y - ex_ctr_y) / ex_height

    target_w = np.log(gt_width / ex_width)
    target_h = np.log(gt_height / ex_width)

    target = np.vstack(
        (target_x, target_y, target_w, target_h)
    ).transpose()

    return target


def bbox_transform_inv(anchors, bbox_deltas):
    """
    根据已经得到的anchors 或者 RoIs 再加上bbox_deltas，我们可以推出
    proposals 和 predicted bboxes
    S"""
    if anchors.shape[0] == 0:
        return np.zeros((0, bbox_deltas.shape[1]), dtype=bbox_deltas.dtype)
    anchors = anchors.astype(bbox_deltas.dtype, copy=False)

    width = anchors[:, 2] - anchors[:, 0] + 1.0
    height = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * width
    ctr_y = anchors[:, 1] + 0.5 * height

    dx = bbox_deltas[:, 0::4]
    dy = bbox_deltas[:, 1::4]
    dw = bbox_deltas[:, 2::4]
    dh = bbox_deltas[:, 3::4]

    pred_ctr_x = dx * width[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * height[:, np.newaxis] + ctr_y[:, np.newaxis]

    pred_w = np.exp(dw) * width[:, np.newaxis]
    pred_h = np.exp(dh) * height[:, np.newaxis]

    pred_boxes =np.zeros(bbox_deltas.shape, dtype= bbox_deltas.dtype)

    # x1,y1,x2,y2
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    把得到的bboxes 进行裁剪
    """
    h, w = im_shape
    boxes[:, 0::4] = np.maximum(np.minimum(w - 1, boxes[:,0::4]), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(h - 1, boxes[:,1::4]), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(w - 1, boxes[:,2::4]), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(h - 1, boxes[:,3::4]), 0)

    return boxes
