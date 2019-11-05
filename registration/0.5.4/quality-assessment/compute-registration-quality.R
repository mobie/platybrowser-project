# Command to produce the output points (example)
# /Applications/elastix_macosx64_v4.8/run_transformix.sh -def /Users/tischer/Desktop/0.5.4/quality-assessment/em-fixed-points.txt -out /Users/tischer/Desktop/0.5.4/quality-assessment -tp /Users/tischer/Desktop/0.5.4/transformations/TransformParameters.BSpline10.0.5.4.txt

file <- "/Users/tischer/Desktop/0.5.4/quality-assessment/outputpoints.txt"
data <- read.table(file, sep = "" , header = F , nrows = 100,
                   na.strings ="", stringsAsFactors= F)

# transformed ProSPr ( in mm, thus * 1000 to convert to um )
tx <- data[31] * 1000
ty <- data[32] * 1000
tz <- data[33] * 1000

# manually assigned ProSPr
library("readxl")

file <- "/Users/tischer/Desktop/0.5.4/quality-assessment/manually_matched_points_0.5.4.xlsx"
data <- read_excel( file )

mx = data[1] 
my = data[2] 
mz = data[3]

distance <- sqrt( ( tx - mx )^2 + ( ty - my )^2 + ( tz - mz )^2 )
print( paste( "mean distance [um]:", mean( distance ) ) )
print( paste( "median distance [um]:", median( distance ) ) )

