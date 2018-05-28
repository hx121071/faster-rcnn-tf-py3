
""""factory method for easily getting imdbs by name"""


__sets={}

import network.VGGnet_train
import network.VGGnet_test
import pdb
import tensorflow as tf

#__sets['VGGnet_test'] =network.VGGnet_test()

#__sets['VGGnet_train'] =network.VGGnet_train()


def get_network(name):
    """get a network by name"""
    #if not __sets.has_key(name):
    #   raise KeyError('Unknown dataset: {}'.format(name))

    if name.split('_')[1]=='test':
        return network.VGGnet_test()
    elif name.split('_')[1]=='train':
        return network.VGGnet_train()
    else:
        return  KeyError('Unknow dataset: {}'.format(name))

def list_networks():
    return __sets.keys()
