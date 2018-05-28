from nms_wrapper import nms
import numpy as np
from nms_py3.cpu_nms import cpu_nms

a = np.array([[10,12,20,22,0.95],[11,12,21,23,0.94]],dtype = np.float32)

b = nms(a, thresh = 0.5)
c = cpu_nms(a, thresh = 0.5)
print(b)
print(c)
