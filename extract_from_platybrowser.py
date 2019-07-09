from scripts.export.extract_subvolume import make_cutout


if __name__ == '__main__':
    levels = get_res_level()
    res_string = " ".join("%i: %s" % (lev, res) for lev, res in enumerate(levels))
    scale_help = ("The resolution level for the cutout."
                  "The levels range from 0 to 6 with resolutions in micrometer: "
                  "%s" % res_string)

    parser = argparse.ArgumentParser("Make a cutout from the platynereis EM data-set and save it as tif stack.")
    parser.add_argument("scale_level", type=int, help=scale_help)
    parser.add_argument("lower_corner", type=str, help="Lower corner of the bounding box to cut out")
    parser.add_argument("upper_corner", type=str, help="Upper corner of the bounding box to cut out")
    parser.add_argument("save_file", type=str, help="Where to save the cutout.")
    args = parser.parse_args()

    print("Converting coordinates from physical coordinates %s:%s" % (args.lower_corner,
                                                                      args.upper_corner))
    bb_start = parse_coordinate(args.lower_corner)
    bb_stop = parse_coordinate(args.upper_corner)
    print("to pixel coordinates %s:%s" % (str(bb_start),
                                          str(bb_stop)))

    print("Extracting raw data")
    raw = make_cutout(args.scale_level, bb_start, bb_stop)
    print("Saving data to tif-stack %s" % args.save_file)
    save_tif(raw, args.save_file)
