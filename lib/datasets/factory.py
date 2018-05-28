__sets = {}

import  datasets.pascal_voc

import numpy as np

#这是我电脑的默认路径
DEVKIT_PATH = '/home/hxx/Faster-RCNN_TF/data/VOCdevkit'
for year in ['2007']:
    for split in ['train','val','trainval','test']:
        name = 'voc_{}_{}'.format(year,split)
        __sets[name] = (lambda split = split, year = year:
                    datasets.pascal_voc(split, year, devkit_path=DEVKIT_PATH))

def get_imdb(name):
    if name not in __sets.keys():
        raise KeyError('Unkown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    return __sets.keys()
