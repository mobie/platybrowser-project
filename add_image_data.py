#! /g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/platybrowser-new/bin/python
import argparse
import json
import multiprocessing
import os

import mobie

ROOT = './data'


def add_new_image_data(path, key, image_name,
                       version, bucket_name,
                       resolution, scale_factors, chunks,
                       target, max_jobs):
    mobie.add_image_data(path, key, ROOT, version, image_name,
                         resolution=resolution, scale_factors=scale_factors, chunks=chunks,
                         target=target, max_jobs=max_jobs)

    dataset_folder = os.path.join(ROOT, version)
    xml_in = os.path.join(dataset_folder, 'images', 'local', f'{image_name}.xml')
    xml_out = os.path.join(dataset_folder, 'images', 'remote', f'{image_name}.xml')

    if bucket_name == 'platybrowser':
        auth = 'Anonymous'
    else:
        auth = 'Protected'

    s3_embl = 'https://s3.embl.de'
    path_in_bucket = os.path.join(version, 'images', 'local', f'{image_name}.n5')

    mobie.xml_utils.copy_xml_as_n5_s3(xml_in, xml_out,
                                      s3_embl, bucket_name, path_in_bucket,
                                      authentication=auth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--key', type=str, default='')
    parser.add_argument('image_name', type=str)

    parser.add_argument('version', type=str)
    parser.add_argument('--bucket_name', type=str, default='platybrowser')

    parser.add_argument('resolution', type=str)
    parser.add_argument('--chunks', type=str, default=None)
    parser.add_argument('--scale_factors', type=str, default=None)

    parser.add_argument('--target', type=str, default='local')
    parser.add_argument('--max_jobs', type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()

    resolution = json.loads(args.resolution)
    chunks = args.chunks
    if chunks is None:
        chunks = (64, 64, 64)
    else:
        chunks = json.loads(chunks)
    scale_factors = args.scale_factors
    if scale_factors is None:
        scale_factors = 5 * [[2, 2, 2]]
    else:
        scale_factors = json.loads(scale_factors)

    add_new_image_data(args.path, args.key,
                       args.image_name, args.version,
                       bucket_name=args.bucket_name,
                       resolution=resolution,
                       scale_factors=scale_factors, chunks=chunks,
                       target=args.target, max_jobs=args.max_jobs)
