
"""
The data layer used during training to train a Fast R-CNN network
RoI DataLayer implements a Caffe Python layer.
"""

from faster_rcnn.config import cfg
from roi_data_layer_py3.minibatch import get_minibatch

import numpy as np

class RoIDataLayer(object):
    """
    Faster R-CNN data layer used for training
    """

    def __init__(self, roidb, num_classes):
        """
        Set the roidb to used by this layer
        """
        self._roidb = roidb
        self._num_classes = num_classes
        self._shuffle_roidb_inds()
    def _shuffle_roidb_inds(self):
        """
        Randomly permute the training roidb index
        """
        ## 设置cur来指向现在访问的长度
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """
        Return the roidb indices for the next minibatch
        """
        ## 实际上只需要向后遍历一个即可

        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()
            self._cur = 0

        db_inds = self._perm[self._cur : self._cur+ cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return  db_inds

    def _get_next_minibatch(self):

        ## 先得到index，再得到index对应的roidb,最后再对roidb做进一步处理
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]

        return get_minibatch(minibatch_db, self._num_classes)

    def forward(self):
        blobs = self._get_next_minibatch()

        return blobs
