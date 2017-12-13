#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: hexiang
import numpy as np
import tensorflow as tf

#还有一些必要库需要添加,但是有点麻烦,暂时当做已知来用

DEFAULT_PADDING = 'SAME'

#下面使用了python3装饰器

def layer(op):
    def layer_decorated(self,*args,**kwargs):
        pass
    pass

class Network(object):
    """docstring for Network.
       defination of the network's basic operation"""
    def __init__(self, inputs, trainable=True):
        self.inputs=[]
        #why dict
        self.layers=dict(inputs)
        self.trainable=trainable
        self.setup()

    def setup(self):
        #必须被子类具体化
        raise NotImplementedError('Must be subclassed')


    def load(self,data_path,session,saver,ignore_missing=False):
        if data_path.endwith('.ckpt'):
            saver.restore(session,data_path)
        else:
            data_dict=np.load(data_path).item()
            for key in data_dict:
                with tf.variable_
