import json
import os
from glob import glob


def make_leveled_views(root):
    leveled_view = {"NormalVector": [0.70, 0.56, 0.43]}
    folders = glob(os.path.join(root, "*.*.*"))
    for folder in folders:
        leveled_view_file = os.path.join(folder, 'misc', 'leveling.json')
        with open(leveled_view_file, 'w') as f:
            json.dump(leveled_view, f)


if __name__ == '__main__':
    root = '../data'
    make_leveled_views(root)
