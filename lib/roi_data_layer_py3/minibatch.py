"""
Compute minibatch for training a Faster R-CNN network
"""

import numpy as np
import numpy.random as npr
import cv2
from faster_rcnn.config import cfg
from utils_py3.blob_helper import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """
    Given a roidb, construct a minibatch sampled from it.
    """
    ## 对于原来得到的roidb,也就是最原始的roidb，进行进一步处理，
    ## 这样就可以得到网络的输入了
    num_images = len(roidb)
    random_scale_inds  = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)

    # 被注释掉的代码一般都是fast-rcnn中的
    # assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    #         'num_images ({}) must divide BATCH_SIZE ({})'. \
    #         format(num_images, cfg.TRAIN.BATCH_SIZE)
    #
    # rois_per_image = cfg.TRIAN.BATCH_SIZE / num_images
    # fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION \
    #                             * rois_per_image)

    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blob = {'data': im_blob}
    # 下面的一些代码有些也不需要，但是可以看一下

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # 下面我们将创建 gt_boxes
        # 同是要创建 im_info
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        ## 这里一定要注意乘以 im_scale, 也就是这张图片对应的scale
        ## 因为scale变了，相应的坐标也就变了
        gt_boxes[:, :4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

        blob['gt_boxes'] = gt_boxes
        blob['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32
        )
    # else: # not using RPN
    #     # Now, build the region of interest and label blobs
    #     rois_blob = np.zeros((0, 5), dtype=np.float32)
    #     labels_blob = np.zeros((0), dtype=np.float32)
    #     bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    #     bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    #     # all_overlaps = []
    #     for im_i in xrange(num_images):
    #         labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
    #             = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
    #                            num_classes)
    #
    #         # Add to RoIs blob
    #         rois = _project_im_rois(im_rois, im_scales[im_i])
    #         batch_ind = im_i * np.ones((rois.shape[0], 1))
    #         rois_blob_this_image = np.hstack((batch_ind, rois))
    #         rois_blob = np.vstack((rois_blob, rois_blob_this_image))
    #
    #         # Add to labels, bbox targets, and bbox loss blobs
    #         labels_blob = np.hstack((labels_blob, labels))
    #         bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
    #         bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
    #         # all_overlaps = np.hstack((all_overlaps, overlaps))
    #
    #     # For debug visualizations
    #     # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
    #
    #     blobs['rois'] = rois_blob
    #     blobs['labels'] = labels_blob
    #
    #     if cfg.TRAIN.BBOX_REG:
    #         blobs['bbox_targets'] = bbox_targets_blob
    #         blobs['bbox_inside_weights'] = bbox_inside_blob
    #         blobs['bbox_outside_weights'] = \
    #             np.array(bbox_inside_blob > 0).astype(np.float32)

    return blob

def _get_image_blob(roidb, scale_inds):
    """
    Build an input blob from the images in
    the roidb at the specified scales
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size =cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS,
                                        target_size, cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
