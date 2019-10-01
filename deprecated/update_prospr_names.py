import os
import json
from glob import glob


def update_prospr_names(reference_folder):
    with open('../data/images.json') as f:
        im_names = json.load(f)
    prefix = 'prospr-6dpf-1-whole'
    prospr_names = [name for name in im_names if name.startswith(prefix)]
    print(len(prospr_names))

    new_prospr_names = glob(os.path.join(reference_folder, '%s-*.xml' % prefix))
    print(len(new_prospr_names))
    new_prospr_names = [os.path.splitext(name)[0] for name in new_prospr_names]
    new_prospr_names = [os.path.split(name)[1] for name in new_prospr_names]

    new_names = [name for name in im_names if not name.startswith(prefix)]
    print(len(new_names))
    new_names += new_prospr_names
    print(len(new_names))

    # check
    for name in new_names:
        path = os.path.join(reference_folder, name) + '.xml'
        assert os.path.exists(path), path
    print("Check passed")

    with open('../data/images.json', 'w') as f:
        json.dump(new_names, f)


if __name__ == '__main__':
    update_prospr_names('../data/0.5.4/images')
