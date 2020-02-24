import pandas as pd
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def main():
    log_path = snakemake.log[0]
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(filename)s:%(funcName)s:%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(snakemake.output.xyz_dir):
        os.makedirs(snakemake.output.xyz_dir)
        os.makedirs(os.path.join(snakemake.output.xyz_dir, 'AP'))
        os.makedirs(os.path.join(snakemake.output.xyz_dir, 'DV'))

    region_table = pd.read_csv(snakemake.input.region_table, sep='\t')
    xyz_table = pd.read_csv(snakemake.input.xyz_table, sep='\t')
    stats_table = pd.read_csv(snakemake.input.stats_table, sep='\t')

    stats_table = stats_table.drop(columns=['label_id_cell', 'label_id_nucleus'])

    xyz_cut = xyz_table.loc[region_table['vnc'] == 1, :]
    stats_cut = stats_table.loc[region_table['vnc'] == 1, :]

    # AP
    for val in stats_cut.columns:
        plt.figure(figsize=(30, 30))
        plt.plot(xyz_cut['new_y'], stats_cut[val], 'o', ms=3)
        plt.title(val, fontsize=24, y=1)
        plt.xlabel('AP', fontsize=24)
        plt.ylabel(val, fontsize=24)
        plt.savefig(os.path.join(snakemake.output.xyz_dir, 'AP', val + '.png'))
        plt.close()

    # DV
    for val in stats_cut.columns:
        plt.figure(figsize=(30, 30))
        plt.plot(xyz_cut['new_z'], stats_cut[val], 'o', ms=3)
        plt.title(val, fontsize=24, y=1)
        plt.xlabel('DV', fontsize=24)
        plt.ylabel(val, fontsize=24)
        plt.savefig(os.path.join(snakemake.output.xyz_dir, 'DV', val + '.png'))
        plt.close()


if __name__ == '__main__':
    main()
