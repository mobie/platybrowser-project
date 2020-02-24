import pandas as pd


def main():
    gene_table = pd.read_csv(snakemake.input.table, sep='\t')
    just_genes = gene_table.drop(columns=['label_id'])

    # group by all columns to get one group per unique row
    # .ngroup() then gives ids 0, 1, 2 etc... for groups
    # https://stackoverflow.com/questions/51110171/how-to-assign-a-unique-id-to-detect-repeated-rows-in-a-pandas-dataframe
    unique_ids = just_genes.groupby(just_genes.columns.tolist(), sort=False).ngroup()

    # add unique ids to original table so can map back later
    gene_table['unique_id'] = unique_ids
    gene_table.to_csv(snakemake.input.table, index=False, sep='\t')

    # save table with duplicates removed
    just_genes['unique_id'] = unique_ids
    duplicates_removed = just_genes.drop_duplicates()
    duplicates_removed.to_csv(snakemake.output.unique_table, index=False, sep='\t')


if __name__ == '__main__':
    main()
