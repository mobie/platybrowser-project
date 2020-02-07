import os
import numpy as np
import z5py


def get_chunk_stats(path, key):
    with z5py.File(path, 'r') as f:
        ds = f[key]
        n_chunks_tot = ds.number_of_chunks

    ds_path = os.path.join(path, key)
    chunk_sizes = []
    for root, dirs, files in os.walk(ds_path):
        for name in files:
            if name == 'attributes.json':
                continue
            size = os.path.getsize(os.path.join(root, name))
            chunk_sizes.append(size)

    n_chunks_filled = len(chunk_sizes)

    return n_chunks_tot, n_chunks_filled, chunk_sizes


def summarise_chunk_stats(path, key):
    n_chunks, n_filled, sizes = get_chunk_stats(path, key)
    percent_filled = float(n_filled) / n_chunks
    with z5py.File(path, 'r') as f:
        ds = f[key]
        chunk_shape = ds.chunks
    print("Checked dataset with chunk shape", chunk_shape)
    print("Number of existing chunks", n_filled, "/", n_chunks, "(", percent_filled, ")")
    print("Mean chunk size in MB:", np.mean(sizes) / 1.e6, "+-", np.std(sizes) / 1.e6)
    print("Min/max chunk size in MB:", np.min(sizes) / 1.e6, "/", np.max(sizes) / 1.e6)


# TODO update this to be more general
def estimate_chunk_sizes():
    ref_chunk_shape = [128, 128, 128]
    ref_chunk_size = float(np.prod(ref_chunk_shape))
    ref_chunk_mb = 1.0160599442530536
    ref_n_chunks = 1553606.

    start = 64
    stop = 128
    step = 16
    for chunk_len in range(start, stop + step, step):
        print("Chunks: %i^3" % chunk_len)
        rel_size = chunk_len ** 3 / ref_chunk_size
        chunk_mb = ref_chunk_mb * rel_size
        n_chunks = ref_n_chunks / rel_size
        print("Nr. chunks at highest res:", int(n_chunks))
        print("Mean chunk size in MB:", chunk_mb)
        print()


if __name__ == '__main__':
    p = '../data/0.6.5/images/local/sbem-6dpf-1-whole-segmented-cells.n5'
    k = 'setup0/timepoint0/s0'
    summarise_chunk_stats(p, k)
