from faster_rcnn.config import cfg
if cfg.USE_GPU_NMS:
    from nms_py3.gpu_nms import gpu_nms
from nms_py3.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
    if dets.shape[0] == 0:
        return []

    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)
