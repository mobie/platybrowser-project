## Code for bilateral pairs analysis

transform_nuclei_cells_to_prospr_space.py transforms coordinates of both the cells and nuclei to ProSPr space - the
transformation parameters used are the same as those here: https://github.com/platybrowser/platybrowser-backend/tree/master/registration/0.6.3/transformations
Results from this are saved as prospr_space_nuclei_points.csv and prospr_space_cell_points.csv.

This code also needs the results of the midline analysis (https://github.com/platybrowser/platybrowser-backend/tree/master/analysis/midline)

xyz_tolerance_pairs.py - calculates the bounds for bilateral xyz criteria, by calculating differences between 
manually curated pairs of bilateral cells.

all_vs_all_neighbours.py - runs the bilateral pairs analysis for different subsets of morphology statistics

all_vs_all_analysis.py - takes results from all_vs_all_neighbours and calculates cumulative sums i.e. fraction of cells that
find a bilateral partner within so many nearest neighbours.  

all_vs_all_graphs.py - takes results from all_vs_all_analysis.py and produces the final graph shown in figure 3
