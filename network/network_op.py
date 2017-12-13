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
        #pass
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

    #load the weights of pre-trained model
    def load(self,data_path,session,saver,ignore_missing=False):
        if data_path.endwith('.ckpt'):
            saver.restore(session,data_path)
        else:
            data_dict=np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var=tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model"+subkey+" to "+key)
                        except ValueError:
                            print("ignore"+key)
                            if not ignore_missing:
                                raise
    
    def feed(self,*args):
        ##下面这行代码确定必须要传入参数
        assert len(args)!=0
        """这里对inputs的一次清空比较重要"""
        self.inputs = []
        
        for layer in args:
            if isinstance(layer,basestring):
                try:
                    layer=self.layers[layer]
                    print(layer)
                except:
                    print(self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)##也即现在的输入为args中的参数,所对应的字典值
        ##必须要返回self,类似于c++中的this指针
        return self
    
    ##返回特定的层的结果
    def get_output(self,layer):
        try:
            layer= self.layers[layer]
        except KeyError:
            print(self.layers.keys())
            raise KeyError('Unknown layer name fed: %s'%layer)
            
    
    def get_unique_name(self,prefix):
        id_sum=sum(t.startwith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(predix,id_sum)
    
    def make_var(self,name,shape,initializer=None,trainable=True):
        return tf.get_variable(name,shape,initializer=initializer,trainable=trainable)
    
    def validate_padding(self,padding):
        assert paddin in ('SAME','VALID')
        
    @layer
    def conv(self,input,k_h,k_w,c_o,s_h,s_w,name,relu=True ,padding=DEFAULT_PADDING,\
             group=1,trianable=True):
        self.validate_padding(padding)
        c_i=input.get_shape()[-1]
        
        assert c_i%group==0
        assert c_o%group==0
        convolve=lambda i,k : tf.nn.conv2d(i,k,[1,s_h,s_w,1],padding=padding)
        with tf.variable_scope(name) as scope:
            
            init_weights= tf.truncated_normal_initializer(0.0,stddev=0.01)
            init_biases=tf.constant_initializer(0.0)
            kernel=self.make_var('weights',[k_h,k_w,c_i/group,c_o],init_weights,\
                                 trainable)
            biases=self.make_var('biases',[c_o],init_biases,trainable)
            
            if group==1:
                conv=convolve(input,kernel)
            else:
                input_groups=tf.split(3,group,input)
                kernel_groups=tf.split(3,group,kernel)
                output_groups=[convolve(i,k) for i,k in zip(input_groups,kernel_groups)]
                conv=tf.concat(3,output_groups)
            if relu:
                bias=tf.nn.bias_add(conv,biases)
                return tf.nn.relu(bias,name=scope.name)
            return tf.nn.bias_add(conv,biases,name=scope.name)
    
    @layer
    def relu(self,input,name):
        return tf.nn.relu(input,name=name)
    
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
