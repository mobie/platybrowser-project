import os
import json
from glob import glob


def make_name_dict():
    prospr_names = glob('../data/rawdata/prospr/*.tif')
    prospr_names = [os.path.split(name)[1] for name in prospr_names]
    prospr_names = [name.split('--')[0] for name in prospr_names]

    new_names = glob('../data/1.0.0/images/local/prospr-6dpf-1-whole-*.xml')
    new_names = [os.path.split(name)[1] for name in new_names
                 if 'virtual' not in name]
    new_names = [os.path.splitext(name)[0] for name in new_names]
    new_names = ['-'.join(name.split('-')[4:]) for name in new_names]
    assert len(new_names) == len(prospr_names)

    name_dict = {}
    for pname in prospr_names:
        if pname.lower() in new_names:
            name_dict[pname] = pname.lower()
        elif pname.startswith('ENR') or pname.startswith('NOV'):
            enr = '-'.join(pname.split('-')[1:]).lower()
            if enr in new_names:
                name_dict[pname] = enr
            else:
                print("Could not match", pname)
        else:
            reg = 'segmented-' + pname.lower()
            if reg in new_names:
                name_dict[pname] = reg
            else:
                print("Could not match", pname)

    manual_matches = {'PNS': 'segmented-lateralectoderm',
                      'Stomodeum': 'segmented-foregut',
                      'ENR19-FboxLike': 'fxl21',
                      'ENR30-CCG5': 'ccvd'}
    name_dict.update(manual_matches)

    print("Number of matches", len(name_dict))
    assert len(name_dict) == len(prospr_names)
    assert len(set(prospr_names) - set(name_dict.keys())) == 0
    assert len(set(new_names) - set(name_dict.values())) == 0

    with open('prospr_name_dict.json', 'w') as f:
        json.dump(name_dict, f)


if __name__ == '__main__':
    make_name_dict()
