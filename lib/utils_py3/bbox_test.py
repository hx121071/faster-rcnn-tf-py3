from utils.cython_bbox import bbox_overlaps
import numpy  as np
import numpy.random as npr

ex_rois = npr.randn(3,4)
gt_rois = npr.randn(2,4)

overlaps = bbox_overlaps(ex_rois, gt_rois)
print(overlaps)
