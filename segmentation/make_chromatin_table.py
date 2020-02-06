# This is kimberly's function to generate the chromatin table. I am putting it here for safe-keeping
# should refactor it to some other place and make it run when scripts are cleaned up
def generate_chromatin_table(root_nuclei_table, out_path, explore=False, intensity=None, label=None):
    """
    Generate chromatin table of xyz positions, chromatin label id, nucleus id & heterochromatin or
    euchromatin status

    :param root_nuclei_table: string, path to root nuclei table (with xyz positions etc)
    :param out_path: string, path to where to save result
    :param explore: whether to add columns for Explore Object Tables
    :param intensity: string, path to intensity image (optional)
    :param label: string, path to label image (optional)
    """
    table_nuclei = pd.read_csv(root_nuclei_table, sep='\t')

    table_label_xyz = table_nuclei[
        ['label_id', 'anchor_x', 'anchor_y', 'anchor_z', 'bb_min_x', 'bb_min_y', 'bb_min_z', 'bb_max_x', 'bb_max_y',
         'bb_max_z']]

    # remove zero label if it exists
    table_label_xyz = table_label_xyz.loc[table_label_xyz['label_id'] != 0, :]

    # rename the label id to nucleus id
    table_label_xyz = table_label_xyz.rename(index=str, columns={"label_id": "nucleus_id"})

    # produce list of all labels i.e. all nucleus ids, then all nucleus ids + 12000
    labels = list(table_label_xyz['nucleus_id'])
    labels2 = [label + 12000 for label in labels]
    result_labels = labels + labels2
    result_labels = [int(label) for label in result_labels]

    # join two copies of table on top of eachother
    result = pd.concat([table_label_xyz, table_label_xyz], axis=0)

    # add the column for the label id
    result['label_id'] = result_labels

    # add column for euchromatin
    euchromatin = [1] * len(labels) + [0] * len(labels2)
    result['euchromatin'] = euchromatin

    # add column for heterochromatin
    heterochromatin = [1 - val for val in euchromatin]
    result['heterochromatin'] = heterochromatin

    # add columns for explore object tables
    if explore:
        result['Path_IntensityImage'] = intensity

        result['Path_LabelImage'] = label

    result.to_csv(out_path, index=False, sep='\t')
