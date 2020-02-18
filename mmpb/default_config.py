import os
import sys
import json
from cluster_tools.cluster_tasks import BaseClusterTask

DEFAULT_GROUP = os.environ.get('PLATYBROWSER_GROUP', 'kreshuk')
DEFAULT_SHEBANG = os.environ.get('PLATYBROWSER_SHEBANG', os.path.realpath(sys.executable))
DEFAULT_BLOCK_SHAPE = [64, 512, 512]


#
# default group parameter
#

def set_default_group(group):
    global DEFAULT_GROUP
    DEFAULT_GROUP = group


def get_default_group():
    return DEFAULT_GROUP


#
# default shebang parameter
#

def set_default_shebang(shebang):
    global DEFAULT_SHEBANG
    DEFAULT_SHEBANG = shebang


def get_default_shebang():
    return DEFAULT_SHEBANG


#
# default block_shape parameter
#

def set_default_block_shape(block_shape):
    global DEFAULT_BLOCK_SHAPE
    DEFAULT_BLOCK_SHAPE = block_shape


def get_default_block_shape():
    return DEFAULT_BLOCK_SHAPE


def write_default_global_config(config_folder):
    os.makedirs(config_folder, exist_ok=True)
    global_config = BaseClusterTask.default_global_config()
    global_config['shebang'] = get_default_shebang()
    global_config['block_shape'] = get_default_block_shape()
    global_config['group'] = get_default_group()
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)
