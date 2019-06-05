import csv


def write_csv(output_path, data, col_names):
    assert data.shape[1] == len(col_names), "%i %i" % (data.shape[1],
                                                       len(col_names))
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(col_names)
        writer.writerows(data)
