import argparse
from paintera_tools import convert_to_paintera_format, set_default_block_shape


def to_paintera():
    path = './data.n5'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    name = args.name

    if name == 'cells':
        pass
    elif name == 'nuclei':
        pass
    elif name == 'cilia':
        pass
    else:
        raise ValueError("Name %s is not supported %s" % name)

    to_paintera()
