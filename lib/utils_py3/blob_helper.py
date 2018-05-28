"""
Blob helper functions
"""

import numpy as np
import cv2

def im_list_to_blob(processed_ims):
    """
    实际上这里做的就是对要输入的im进行一个统一规格
    """
    # print(processed_ims[0].shape)
    max_shape = np.array([im.shape for im in processed_ims]).max(axis=0)
    # print("max shape is:",max_shape)
    num_images = len(processed_ims)

    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                     dtype=np.float32)

    for i in range(num_images):
        im = processed_ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """
    Mean subtract and scale an image for use in a blob
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
