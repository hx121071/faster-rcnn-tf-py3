import os
import os.path as osp
import numpy as np
from distutils import spawn
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# rpn cfg

__C.TRAIN = edict()
# get data blob
__C.TRAIN.SCALES = [600,]
__C.TRAIN.IMS_PER_BATCH = 1
__C.TRAIN.MAX_SIZE = 1000

# add flipped image

__C.TRAIN.USE_FLEPPED =True

# anchor_target_layer params
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
__C.TRAIN.RPN_FG_FRACTION = 0.5
__C.TRAIN.RPN_BATCHSIZE = 256
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1

# proposal_layer params
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
__C.TRAIN.RPN_MIN_SIZE = 16
__C.TRAIN.RPN_NMS_THRESH = 0.7

# proposal_target_layer params
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.FG_FRACTION = 0.25
__C.TRAIN.FG_THRESH = 0.5

__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# If an anchor statisfied by positive and negative conditions set to negative

__C.TRAIN.RPN_CLOBBER_POSITIVES = False

__C.TRAIN.HAS_RPN = True


## train.py 中需要的
## bbox normalization
## 我在程序中唯一没看懂的一个点

__C.TRAIN.BBOX_REG = True

# learning rate
__C.TRAIN.LEARNING_RATE = 0.001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = 50000
__C.TRAIN.DISPLAY = 1
__C.IS_MULTISCALE = False

# Enable timeline generation
__C.TRAIN.DEBUG_TIMELINE = False
## 保存名字

__C.TRAIN.SNAPSHOT_PREFIX = 'VGGnet_fast_rcnn'
__C.TRAIN.SNAPSHOT_INFIX = ''

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000


########## test options ##########
__C.TEST = edict()

__C.TEST.SCALES = (600,)

__C.TEST.MAX_SIZE = 1300

__C.TEST.NMS = 0.3

__C.TEST.SVM = False

__C.TEST.BBOX_REG = True

__C.TEST.HAS_RPN = True


__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
#__C.TEST.RPN_PRE_NMS_TOP_N = 12000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
#__C.TEST.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

# Enable timeline generation
__C.TEST.DEBUG_TIMELINE = False

__C.DEDUP_BOXES = 1./16.

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


if spawn.find_executable("nvcc"):
    # Use GPU implementation of non-maximum suppression
    __C.USE_GPU_NMS = True

    # Default GPU device id
    __C.GPU_ID = 0
else:
    __C.USE_GPU_NMS = False
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

print(__C.ROOT_DIR)

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# A small number that's used many times
__C.EPS = 1e-14

__C.EXP_DIR = 'default'

__C.RNG_SEED = 3


def get_output_dir(imdb, weights_filename):
    """
    Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """

    outdir = osp.abspath(osp.join(__C.ROOT_DIR,
                        'output',
                        __C.EXP_DIR, imdb.name))

    if weights_filename is not None:
        outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """
    Load o config file and merge it into the default options.
    """
    import yaml
    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
