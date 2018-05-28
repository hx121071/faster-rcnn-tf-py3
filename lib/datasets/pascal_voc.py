import os
print(__file__)
from datasets.imdb import imdb
# import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
# import utils.cython_bbox
import pickle
import subprocess
import uuid
# from voc_eval import voc_eval
from faster_rcnn.config import cfg
import pdb

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',\
                        'zebrish')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.config = {
                        'cleanup' : True,\
                        'use_salt' : True,\
                        'use_diff' : False,\
                        'matlab_eval' : False,\
                        'rpn_file' : None,\
                        'min_size' : 2
        }
        assert os.path.exists(self._devkit_path),\
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path),\
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages',\
                                    index + self._image_ext)
        assert os.path.exists(image_path),\
                'Path does not exist: {}'.format(image_path)
        return image_path
    def _load_image_set_index(self):
        """
        Load the indexes listed in this  dataset's image set file.
        """
        # Example path to image set file
        # self._devkit_path + /VOCdevkit2007/VOC2007/Imagesets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',\
                                        self._image_set + '.txt')
        assert os.path.exists(image_set_file),\
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.strip() for x  in f.readlines()]

        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdekit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """

        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                roidb = pickle.load(fid)
            print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)\
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)

        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC format
        """

        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        if not self.config['use_diff']:
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text)==0
            ]
            objs = non_diff_objs

        num_objs = len(objs)

        boxes = np.zeros((num_objs,4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        #Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {
                'boxes' : boxes,\
                'gt_classes' : gt_classes,\
                'gt_overlaps' : overlaps,\
                'flipped' : False,\
                'seg_areas' : seg_areas
        }

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007' , '/home/hxx/Faster-RCNN_TF/data/VOCdevkit')
    res = d.roidb()
    print(res[0].keys())
    print(len(res))
