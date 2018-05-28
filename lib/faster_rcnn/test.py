from faster_rcnn.config import cfg, get_output_dir
import argparse
from utils_py3.timer import Timer

import numpy as np
import cv2
from utils_py3.cython_nms import nms
# from utils_py3.boxes_grid import get_boxes_grid

import pickle
# import heapq
from utils_py3.blob_helper import im_list_to_blob
import os
import math
import tensorflow as tf
from faster_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
import time
import pdb


def _get_image_blob(im):
    """
    Convert an image into a network input.
    Argument :
        im(ndarray): a color image in BGR order

    Returns :
        blob(ndarray): a data blob holding an image pyramid
        im_scales_factors(list):list of image scales (relative to im) used
             in the image pyramid
    """

    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    print("im_shape is:",im_shape)
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    print("min and max:",im_size_min,im_size_max)
    processed_ims = []
    im_scale_factors = []

    for target_size in  cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

# def _project_im_rois(im_rois, scales):
#     """Project image RoIs into the image pyramid built by _get_image_blob.
#     Arguments:
#         im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
#         scales (list): scale factors as returned by _get_image_blob
#     Returns:
#         rois (ndarray): R x 4 matrix of projected RoI coordinates
#         levels (list): image pyramid levels used by each projected RoI
#     """
#     im_rois = im_rois.astype(np.float, copy=False)
#     scales = np.array(scales)
#
#     if len(scales) > 1:
#         widths = im_rois[:, 2] - im_rois[:, 0] + 1
#         heights = im_rois[:, 3] - im_rois[:, 1] + 1
#
#         areas = widths * heights
#         scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
#         diff_areas = np.abs(scaled_areas - 224 * 224)
#         levels = diff_areas.argmin(axis=1)[:, np.newaxis]
#     else:
#         levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
#
#     rois = im_rois * scales[levels]
#
#     return rois, levels

def _get_blobs(im):
    """
    Convert an image and RoIs within that image into  inputs
    """
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return  blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries
    """

    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)

    return  boxes


def _recales_boxes(boxes, inds, scales):
    """
    Rescale boxes according to image rescaling
    """
    for  i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return  boxes

def im_detect(sess, net, im, boxes=None):
    """
    Detect object classes in an image given object proposals
    Arguments:
        net: faster rcnn network to use
        im: color image to test(in BGR order)
        boxes(ndarray): R X 4 array of object proposals
    Returns:
        scores(ndarray): R X K array of object class scores(K includes
                background as object category 0)
        boxes(ndarray): R X (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im)

    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[1],
                                  im_blob.shape[2],
                                  im_scales[0]]], dtype=np.float32)
    print(blobs['im_info'])
    feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'],
                net.keep_prob: 1.0}

    run_options = None
    run_metadata = None
    if cfg.TEST.DEBUG_TIMELINE:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    # pdb.set_trace()
    cls_score, cls_prob, bbox_pred, rois = sess.run(
        [net.get_output('cls_score'), net.get_output('cls_prob'), net.get_output('bbox_pred'),
         net.get_output('rois')], feed_dict = feed_dict, options=run_options,run_metadata=run_metadata
    )
    assert len(im_scales) == 1, "Only single-image batch implemented"
    boxes = rois[:, 1:5] / im_scales[0]

    scores = cls_prob

    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)

    if cfg.TEST.DEBUG_TIMELINE:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open(str(long(time.time() * 1000)) + '-test-timeline.ctf.json', 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
        trace_file.close()

    return scores,pred_boxes

def vis_detections(im, class_name, dets, thresh=0.8):
    """
    Visual debugging of detections
    """

    import matplotlib.pyplot as plt

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.gca().text(bbox[0], bbox[1] - 2,
                 '{:s} {:.3f}'.format(class_name, score),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')

            plt.title('{} {:.3f}'.format(class_name, score))

# def apply_nms(all_boxes, thresh):

def test_net(sess, net, imdb, weights_filename, max_per_image=300,
             thresh=0.05, vis=False):

    """
    Test a faster rcnn network on an image database
    """

    num_images = len(imdb.image_index)
    # pdb.set_trace()
    all_boxes = [[[] for _ in  range(num_images)]
                  for _  in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)

    _t = {'im_detect': Timer(), 'misc': Timer()}
    for i in range(2):
        box_proposals = None

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im, box_proposals)
        _t['im_detect'].toc()
        # pdb.set_trace()
        _t['misc'].tic()

        if vis:
            image = im[:, :, (2, 1, 0)]
            plt.cla()
            plt.imshow(image)

        # skip j = 0, because it's the background class

        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)

            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets =  cls_dets[keep, :]
            print(cls_dets)
            if vis:
                vis_detections(image, imdb.classes[j], cls_dets)

            all_boxes[j][i] = cls_dets
        if vis:
            plt.show()

        # Limit to max_per_image detections * over all classes *
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                     for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                if j in range(1, imdb.num_classes):
                    keep = np.where([all_boxes[j][i][:, -1]] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        _t['misc'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)


    # print ('Evaluating detections')
    # imdb.evaluate_detections(all_boxes, output_dir)
