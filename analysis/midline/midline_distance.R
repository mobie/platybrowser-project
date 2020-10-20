library(tidyverse)
library(rgl)

# read in manually chosen points on midline
points = read.table(
    "Z:\\Kimberly\\Projects\\SBEM_analysis\\src\\sbem_analysis\\paper_code\\files_for_midline_xyz\\points_for_midline.txt", sep='\t',
    header=FALSE)
names(points) <- c('x', 'y', 'z')

# plot them to check they look ok
rgl::plot3d(points, size=6)
rgl::aspect3d("iso")
rgl.close()

# first I tried fitting a polynomial of the form z ~ poly(x, y, degree = 2), but the plane is near vertical and
# hard to fit. Flipping it to instead match x with x ~ poly(y, z) gave much better results

# fit surface

# use linear model function, degree being the degree of the polynomial
# here I fit a second order polynomial of two variables
# general form of second order is ay**2 + bz**2 + cyz + dy + ez + intercept
x <- points$x
y <- points$y
z <- points$z
fit <- lm(x ~ poly(y, z, degree=2, raw=TRUE), data = points)

# summary(fit) can give you coefficients in the Estimate column
# so i.e. x = ay**2 + bz**2 + cyz + dy + ez + intercept

# predict values over a grid that covers the original points to take a look at the fit
# could do closer spacing to increase accuracy of distances later
y <- min(y): max(y)
z <- min(z): max(z)
grid <- expand.grid(y, z)
names(grid) <-  c("y", "z")

# predictd x values
vals <- predict(fit, newdata=grid)

# results
result <- cbind(grid, vals)
names(result)[3] <- 'x'

# clip x to make plotting easier to view
result <- result[result$x < max(x),]
result <- result[result$x > min(x),]

# plot the points and the fitted surface
rgl::plot3d(points[, c('x', 'y', 'z')], size = 6, axis.scales = FALSE, col = 'red', xlim = c(min(x), max(x)), ylim = c(
    min(y), max(y)), zlim = c(min(z), max(z)))
rgl::points3d(result[, c('x', 'y', 'z')], size = 6, axis.scales = FALSE, col = 'black', alpha = 0.2)
rgl.close()

# now want to add distances of nuclei from this midline
# fit over a grid that covers the whole image
# could do closer spacing to increase accuracy of distances later (currently grid spacing is every micron)

# 0 to max dimension in microns (from xml files)
y <- 0: 260
z <- 0: 290
grid <- expand.grid(y, z)
names(grid) <-  c("y", "z")

# predictd x values
vals <- predict(fit, newdata=grid)

# results
result <- cbind(grid, vals)
names(result)[3] <- 'x'

# plot the points and the fitted surface
rgl::plot3d(points[, c('x', 'y', 'z')], size = 6, axis.scales = FALSE, col = 'red', ylim = c(min(y), max(y)), zlim = c(
    min(z), max(z)))
rgl::points3d(result[, c('x', 'y', 'z')], size = 6, axis.scales = FALSE, col = 'black', alpha = 0.2)
rgl::aspect3d("iso")
rgl.close()

# read in nucleus dataframe
nuclei = read.table('W:\\EM_6dpf_segmentation\\platy-browser-data\\data\\1.0.1\\tables\\sbem-6dpf-1-whole-segmented-nuclei\\default.csv', sep='\t',
                    header=TRUE)
# remove zero label if present
nuclei <- nuclei[nuclei$label_id != 0,]
# get xyz coords
nuclei_cut <- nuclei[, c('anchor_x', 'anchor_y', 'anchor_z')]
result <- result[, c('x', 'y', 'z')]

# same for cells
cells = read.table('W:\\EM_6dpf_segmentation\\platy-browser-data\\data\\1.0.1\\tables\\sbem-6dpf-1-whole-segmented-cells\\default.csv', sep='\t',
                   header=TRUE)
cells <- cells[cells$label_id != 0,]
# get xyz coords
cells_cut <- cells[, c('anchor_x', 'anchor_y', 'anchor_z')]

# point on midline - randomly chosen one near the centre if platy
midline_point <- c(163.8045, 149.22421, 141.440733)
# nucleus on one side - randomly chosen as near centre of platy
# label id 8290 (in version 1.0.0)
nucleus_point <- c(107.619, 163.189, 155.880)

# vector from midline to nucleus - will use this to assess which side of the midline the rest of the nuclei are on
mid_vec <- nucleus_point - midline_point

get_dists <- function(table, result) {
  # loop through each nucleus and calculate the minimum distance between it and one of the predicted points on the midline
  # (again limited by grid spacing - here a micron). For more accuracy either use a finer grid, or figure out a way to properly calculate
  # the distances from the equation of the surface, or perhaps mesh it and use a package to figure out point to mesh distance?
  dists <- apply(table, 1, function(xyz)
  {
    
    # dataframe of just the same nucleus over and over
    xyz_data <- matrix(rep(xyz, nrow(result)), nrow=nrow(result), ncol=3, byrow=TRUE)
    
    # calculate x**2 + y**2 + z**2
    t <- rowSums((result - xyz_data) ** 2)
    # square root to get distance
    t <- sqrt(t)
    index <- which.min(t)
    
    # coordinate on midline that gave minimum distance
    min_coord <- result[index,]
    
    # vector from midline to nucleus
    point_vec <- xyz - min_coord
    
    # dot product of this with midline point to nucleus on chosen side before
    dot_prod <- sum(point_vec * mid_vec)
    
    # if larger than zero, both nuclei used in the vectors are on the same side of the midline
    # otherwise opposite
    # indicate this with the sign of the result
    if (dot_prod > 0)
    {
      return (min(t))
      
    } else {
      return (-min(t))
    }
    
  })
  
  # add it to the table
  table$distances = dists
  
  # boolean for which side nucleus is on
  table$side = table$distances > 0
  table$side2 = table$distances < 0
  
  # absolute distance values
  table$absolute = abs(table$distances)
  
  return(table)
  
}


nuclei_cut <- get_dists(nuclei_cut, result)
nuclei_cut <- nuclei_cut[,c('distances', 'side', 'side2', 'absolute')]
nuclei <- cbind(nuclei, nuclei_cut)
write.table(nuclei, "Z:\\Kimberly\\Projects\\SBEM_analysis\\src\\sbem_analysis\\paper_code\\files_for_midline_xyz\\distance_from_midline_nuclei_1_0_1.csv",
            sep='\t', quote=FALSE, row.names = FALSE)

cells_cut <- get_dists(cells_cut, result)
cells_cut <- cells_cut[,c('distances', 'side', 'side2', 'absolute')]
cells <- cbind(cells, cells_cut)
write.table(cells, "Z:\\Kimberly\\Projects\\SBEM_analysis\\src\\sbem_analysis\\paper_code\\files_for_midline_xyz\\distance_from_midline_cells_1_0_1.csv",
            sep='\t', quote=FALSE, row.names = FALSE)