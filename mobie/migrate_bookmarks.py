import json
import os
from glob import glob

from utils import update_dict


def migrate_bookmarks(misc_folder):
    bookmark_folder = os.path.join(misc_folder, 'bookmarks')
    os.makedirs(bookmark_folder, exist_ok=True)

    # make the default bookmark
    default_bookmark_path = os.path.join(bookmark_folder, 'default.json')
    default_bookmark = {
        "default": {
            "layers": {
                "sbem-6dpf-1-whole-raw": {
                    "contrastLimits": [0., 255.]
                }
            }
        }
    }
    with open(default_bookmark_path, 'w') as f:
        json.dump(default_bookmark, f, indent=2, sort_keys=True)

    # copy the paper bookmarks
    old_bookmark_path = os.path.join(bookmark_folder, 'bookmarks.json')
    with open(old_bookmark_path) as f:
        bookmarks = json.load(f)

    bookmarks = update_dict(bookmarks)

    new_bookmark_path = os.path.join(misc_folder, 'manuscript_bookmarks.json')
    with open(new_bookmark_path, 'w') as f:
        json.dump(new_bookmark_path, f, indent=2, sort_keys=True)

    # TODO
    # subprocess.run(['git', 'rm', old_bookmark_path])


def migrate_all_bookmakrs(root):
    folders = glob(os.path.join(root, "*.*"))
    for folder in folders:
        misc_folder = os.path.join(folder, 'misc')
        migrate_bookmarks(misc_folder)


if __name__ == '__main__':
    root = '../data'
    migrate_all_bookmakrs(root)
