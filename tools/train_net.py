import _init_paths
from faster_rcnn.train import get_training_roidb, get_data_layer, train_net
from faster_rcnn.config import cfg,cfg_from_file,cfg_from_list,get_output_dir
from datasets.factory import get_imdb
from network.factory import get_network
import argparse
import pprint
import numpy as  np
import sys
import pdb
import cv2


def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description = 'Train a Faster R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver', help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained_model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='random (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of network',
                        type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(arg.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        np.random.seed(cfg.RNG_SEED)

    imdb = get_imdb(args.imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))

    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print(device_name)

    network = get_network(args.network_name)
    print('Use network `{:s}` in training'.format(args.network_name))

    # data_layer = get_data_layer(roidb, imdb.num_classes)
    #
    # blob = data_layer.forward()
    # # cv2.imshow("test", blob['data'][0, :, :, :])
    # cv2.imshow("test",blob['data'][0].astype(np.uint16))
    # cv2.waitKey(0)
    # print(blob['gt_boxes'])
    train_net(network, imdb, roidb, output_dir,
              pretrained_model = args.pretrained_model,
              max_iters  = args.max_iters)
