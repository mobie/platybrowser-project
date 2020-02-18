import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_nephr_ids(version):
    table_path = '../../data/%s/tables/sbem-6dpf-1-whole-segmented-cells/regions.csv' % version
    table = pd.read_csv(table_path, sep='\t')
    nephr_ids = table['nephridia'].values
    right_nephr_ids = np.where(nephr_ids == 1)[0]
    left_nephr_ids = np.where(nephr_ids == 2)[0]
    return right_nephr_ids, left_nephr_ids


def check_cell_ids(version):
    table_path = '../../data/%s/tables/sbem-6dpf-1-whole-segmented-cilia/cell_mapping.csv' % version

    right_nephr_ids, left_nephr_ids = get_nephr_ids(version)
    table = pd.read_csv(table_path, sep='\t')
    cell_ids = table['cell_id'].values

    matched_right = []
    matched_left = []
    for cell_id in cell_ids:
        if cell_id in right_nephr_ids:
            matched_right.append(cell_id)
        elif cell_id in left_nephr_ids:
            matched_left.append(cell_id)
        else:
            assert cell_id == 0 or np.isnan(cell_id), str(cell_id)

    matched_right = set(matched_right)
    matched_left = set(matched_left)

    # unmatched_right = set(right_nephr_ids) - matched_right
    # unmatched_left = set(left_nephr_ids) - matched_left

    print("Number of cells right:")
    print("Total:", len(right_nephr_ids))
    print("With cilia:", len(matched_right))

    print("Number of cells left:")
    print("Total:", len(left_nephr_ids))
    print("With cilia:", len(matched_left))


def get_count_data(counts_left, counts_right):
    n_cells = len(counts_left)
    cell_id = list(range(n_cells)) + list(range(n_cells))
    side = n_cells * ["Left"] + n_cells * ["Right"]
    counts = counts_left + counts_right

    data = np.array([cell_id, side, counts]).T
    data = pd.DataFrame(data, columns=["Cells", "Side", "Number of Cilia"])
    return data


def plot_cilia_per_cell(version):
    counts_left = []
    counts_right = []

    table_path = '../../data/%s/tables/sbem-6dpf-1-whole-segmented-cilia/cell_mapping.csv' % version
    table = pd.read_csv(table_path, sep='\t')
    cell_ids = table['cell_id']
    cell_ids = cell_ids[cell_ids != 0]
    cell_ids = cell_ids[~np.isnan(cell_ids)]
    cell_ids = cell_ids.astype('uint32')

    right_nephr_ids, left_nephr_ids = get_nephr_ids(version)
    unique_cell_ids = np.unique(cell_ids)

    total_count = 0
    for cell_id in unique_cell_ids:
        n_cilia = np.sum(cell_ids == cell_id)
        total_count += n_cilia
        if cell_id in right_nephr_ids:
            counts_right.append(n_cilia)
        elif cell_id in left_nephr_ids:
            counts_left.append(n_cilia)
        else:
            print(cell_id)

    counts_right.sort(reverse=True)
    counts_left.sort(reverse=True)
    assert len(counts_right) == len(counts_left)
    print("Total number of cilia:", total_count)
    print("Right counts:")
    print(sum(counts_right))
    print("Left counts:")
    print(sum(counts_left))

    count_data = get_count_data(counts_left, counts_right)

    sns.set(style="whitegrid", font="Arial", font_scale=1.4)
    ax = sns.barplot(x="Cells", y="Number of Cilia", hue="Side", data=count_data)
    ax.set_title("Cilia distribution of nephridia cells")
    ax.set_xlabel('')
    ax.set_xticks([])
    plt.show()


if __name__ == '__main__':
    version = '0.6.5'
    # check_cell_ids(version)
    plot_cilia_per_cell(version)
