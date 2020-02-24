import numpy as np
import logging
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_clusters_on_umap(umap, clusters, label=True, assignment=None, palette='hls', alpha=0.8, prefix='c',
                          fontsize=20):
    """
    Plot specified clusters on UMAP, groups all clusters with only one member into cluster called 'single' (just
    to keep the plot readable)
    :param umap: umap, 2 columns
    :param clusters: list of ints, cluster assignment for each row of the umap (length should be == nrow in umap)
    :param label: boolean, whether to add text for clusters or not
    :param prefix: prefix to add to name of cluster
    :param assignment: dictionary, if you want to assign clusters specific names e.g. 'muscles' etc can pass a
    dictionary like e.g. assignemnt = {'c1':'muscles', 'c2':'head'} Note at the moment I'm using numpy string arrays
    with a max number of characters of 30, passing longer labels will result in truncation
    :param palette: colour palette to use for plotting
    :param fontsize: font size to use
    """

    # make categorical so graph will colour properly
    # can't just be '1' etc, as seaborn will misinterpret this as not being categorical
    # numpy arrays need to be given the max string length, here allow up to 30 characters
    clusters_current = np.array([prefix + str(val) for val in clusters], dtype="<U30")

    # unique clusters
    cluster_ids, counts = np.unique(clusters_current, return_counts=True)

    # print number of clusters produced
    no_clusters = len(cluster_ids)
    logger.info("No. of clusters: " + str(no_clusters))

    # find clusters with only one member
    to_dump = cluster_ids[counts <= 1]
    no_singletons = len(to_dump)
    logger.info("No. of singleton clusters: " + str(no_singletons))

    if len(to_dump) > 0:
        clusters_current[np.isin(clusters_current, to_dump)] = 'single'
        cluster_ids = np.unique(clusters_current)

    plt.figure(figsize=(30, 20))
    sns.set()
    sns.set_style('white')
    plot = sns.scatterplot(x=umap[:, 0], y=umap[:, 1], hue=clusters_current, palette=palette, alpha=alpha, linewidth=0)

    if label:

        # add text annotations of cluster

        # add mapping if provided
        if assignment is not None:
            for key, value in assignment.items():
                clusters_current[clusters_current == key] = value
                cluster_ids = np.unique(clusters_current)

        if len(to_dump) > 0:
            clusters_to_label = cluster_ids[cluster_ids != 'single']
        else:
            clusters_to_label = cluster_ids
        x_coords = [statistics.median(umap[:, 0][clusters_current == clust]) for clust in clusters_to_label]
        y_coords = [statistics.median(umap[:, 1][clusters_current == clust]) for clust in clusters_to_label]

        for x, y, lab in zip(x_coords, y_coords, clusters_to_label):
            plot.text(x, y, lab, horizontalalignment='left', size='medium', color='black', weight='semibold',
                      fontsize=fontsize)

    return plot, no_clusters, no_singletons


def plot_continuous_feature_on_umap(umap, column, palette='Blues'):
    """
    Plot a certain column (containing continuous float values) on the UMAP

    :param umap: umap, 2 columns
    :param column: list of floats, values for each row of the umap (length should be == nrow in umap)
    :param palette: colour palette to use
    """

    plt.figure(figsize=(30, 20))
    sns.set_context()
    sns.set_style('white')
    plot = sns.scatterplot(x=umap[:, 0], y=umap[:, 1], hue=column, palette=palette, alpha=0.8, linewidth=0)

    return plot


def plot_categorical_feature_on_umap(umap, column, label=False):
    """
        Plot a certain column (containing categorical int values) on the UMAP

        :param umap: umap, 2 columns
        :param column: list of ints, values for each row of the umap (length should be == nrow in umap)
        """

    # coerce to int if not
    column = column.astype(int)

    no_classes = len(np.unique(column))

    # skip if all one type
    if no_classes < 2:
        logger.info('Only one category!!')
        return

    if no_classes == 2:
        plt.figure(figsize=(30, 20))
        sns.set_context()
        sns.set_style('white')
        plot = sns.scatterplot(x=umap[:, 0], y=umap[:, 1], hue=column, palette=['#999999', '#000065'], alpha=0.8,
                               linewidth=0)

    else:

        plt.figure(figsize=(30, 20))
        sns.set_context()
        sns.set_style('white')
        plot = sns.scatterplot(x=umap[:, 0], y=umap[:, 1], hue=column, palette=sns.color_palette('Paired', no_classes),
                               alpha=0.8, linewidth=0, legend='full')

        if label:
            region_ids = np.unique(column)
            x_coords = [statistics.median(umap[:, 0][column == reg]) for reg in region_ids]
            y_coords = [statistics.median(umap[:, 1][column == reg]) for reg in region_ids]

            for x, y, lab in zip(x_coords, y_coords, region_ids):
                plot.text(x, y, lab, horizontalalignment='left', size='medium', color='black', weight='semibold',
                          fontsize=20)

    return plot


#  as here: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def silhouette_plot(X, cluster_labels, result_file):
    plt.figure(figsize=(20, 20))

    n_clusters = len(np.unique(cluster_labels))
    cluster_labels = np.array(cluster_labels)

    # The 1st subplot is the silhouette plot
    plt.xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(X) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(X, cluster_labels)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.savefig(result_file)
    plt.close()
