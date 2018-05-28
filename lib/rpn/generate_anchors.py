import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:

# anchors =
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                      scales=2**np.arange(3,6)):


    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors  = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scales_enum(ratio_anchors[i,:], scales) \
                        for i in range(ratio_anchors.shape[0])])
    return anchors
def _whctr(base_anchor):
    w = base_anchor[2] - base_anchor[0] + 1
    h = base_anchor[3] - base_anchor[1] + 1
    x_ctr = base_anchor[0] + 0.5 * (w - 1)
    y_ctr = base_anchor[1] + 0.5 * (h - 1)

    return  w,h,x_ctr,y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return  anchors
def _ratio_enum(base_anchor, ratios):

    w,h,x_ctr,y_ctr = _whctr(base_anchor)
    size = w*h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scales_enum(anchor, scales):

    w, h, x_ctr, y_ctr = _whctr(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return  anchors
if __name__ == '__main__':
    anchors = generate_anchors()
    print(anchors)
