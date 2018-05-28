from rpn.proposal_layer_tf import proposal_layer
import numpy as np

rpn_cls_prob =[[[[0.95, 0.85],[0.75, 0.76]] for i in range(18)]]
rpn_bbox_pred = np.ones((1,2,2,36), dtype=np.float32)

rpn_cls_prob = np.array(rpn_cls_prob).astype(np.float32).transpose(0,2,3,1)

im_info =[[600, 800, 1.0]]
cfg_key = 'TRAIN'

# print(rpn_cls_prob.shape)
blob = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key)

print(blob)
