import argparse
import os
import pandas as pd
from scripts.export.extract_subvolume import make_cutout, get_res_level, parse_coordinate, name_to_path


def get_bounding_box(tag, lower, upper, table_name, table_id):

    # validate the inputs
    have_coordinates = lower and upper
    have_table = table_name and table_id is not None

    assert have_coordinates != have_table, "Need one of coordinates or table"
    if have_coordinates:
        print("Converting coordinates from physical coordinates %s:%s" % (lower,
                                                                          upper))
        bb_start = parse_coordinate(lower)
        bb_stop = parse_coordinate(upper)
        print("to pixel coordinates %s:%s" % (str(bb_start),
                                              str(bb_stop)))
    else:
        name = os.path.splitext(os.path.split(name_to_path(table_name))[1])[0]
        table_path = os.path.join('data', tag, 'tables', name, 'base.csv')
        assert os.path.exists(table_path), "Could not find table for %s at %s" % (table_name,
                                                                                  table_path)
        table = pd.read_csv(table_path, sep='\t')
        sub_table = table.loc[table['label_id'] == table_id]
        assert sub_table.shape[0] == 1
        for row in sub_table.itertuples(index=False):
            bb_start = [row.bb_min_x, row.bb_min_y, row.bb_min_z]
            bb_stop = [row.bb_max_x, row.bb_max_y, row.bb_max_z]
        print("Read bounding box from table %s for id %i:" % (table_name, table_id))
        print(bb_start, ":", bb_stop)
    return bb_start, bb_stop


# TODO
# - expose out_format as argument
# - saving multiple names in one go currently only works for h5 and n5, fix this!
if __name__ == '__main__':
    levels = get_res_level()
    res_string = " ".join("%i: %s" % (lev, res) for lev, res in enumerate(levels))
    scale_help = ("The resolution level for the cutout."
                  "The levels range from 0 to 6 with resolutions in micrometer: "
                  "%s" % res_string)

    parser = argparse.ArgumentParser("Make a cutout from the platynereis EM data-set.")
    # Mandatory arguments: which tag and scale
    parser.add_argument("tag", type=str, help="Version tag of the data to extract from")
    parser.add_argument("scale_level", type=int, help=scale_help)

    # Optional: data names
    parser.add_argument("--names", type=str, nargs='+', default=['raw'],
                        help=("Names of the data to be extracted.\
                              Choose from 'raw', 'cells', 'nuclei', 'chromatin', 'cilia'"))

    # Optional: select bounding box from platy browser coordinates
    parser.add_argument("--lower_corner", type=str, default='',
                        help="Lower corner of the bounding box to cut out",)
    parser.add_argument("--upper_corner", type=str, default='',
                        help="Upper corner of the bounding box to cut out")

    # Optional: Select bounding box from table and id
    parser.add_argument('--table_name', type=str, default='',
                        help="Table from where to read the bounding box, can be from the same option as names")
    parser.add_argument('--table_id', type=int, default=None,
                        help="Id for bounding box")

    # Optional: select where to save the data
    parser.add_argument("--save_file", type=str, help="Where to save the cutout.",
                        default='')
    args = parser.parse_args()

    # 1.) validate the tag
    tag = args.tag
    folder = os.path.join('data', tag)
    assert os.path.exists(folder), "Invalid tag %s" % tag

    # 2.) figure out the bounding box
    lower = args.lower_corner
    upper = args.upper_corner
    table_name = args.table_name
    table_id = args.table_id
    bb_start, bb_stop = get_bounding_box(tag, lower, upper, table_name, table_id)

    # 3.) extract data for all names
    names = args.names
    save_file = args.save_file
    scale = args.scale_level
    for name in names:
        print("Extracting data for %s" % name)
        make_cutout(tag, name, scale, bb_start, bb_stop, save_file)
