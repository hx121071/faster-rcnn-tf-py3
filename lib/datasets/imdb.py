import os
import os.path as osp
import PIL
# from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from faster_rcnn.config import cfg

class imdb(object):
    """
    Image database
    """
    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        # self._obj_proposer = 'selective_search'
        self._roidb = None
        # print(self.default_roidb)
        self._roidb_handler = self.default_roidb
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val


    @property
    def roidb(self):
        # roidb is a list of dictionaries which has the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not  None:
            return self._roidb

        self._roidb = self._roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self._image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    # test is need
    def evaluate_detections(self, all_boxes, output_dir =None):
        """
        all_boxes is a list of length number-of-classes/
        Each list element is a list of length num-of-images.
        Eacg of those list elements is either an empty list []
        or a numpy array of evaluate_detections

        all_boxes[class][image]  = [] or np.array of shape #dets X 5
        """
        raise NotImplementedError

    def _get_width(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0] \
                for i in range(self.num_images)]

    def append_flipped_images(self):
        num_images = self.num_images
        width = self._get_width()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = width[i] - oldx2 - 1
            boxes[:, 2] = width[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {
                    'boxes': boxes,\
                    'gt_overlaps': self.roidb[i]['gt_overlaps'],\
                    'gt_classes': self.roidb[i]['gt_classes'],\
                    'flipped': True
            }
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
