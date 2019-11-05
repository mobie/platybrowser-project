# Command to produce the output points (example)
# /Applications/elastix_macosx64_v4.8/run_transformix.sh -def /Users/tischer/Desktop/9.9.9/quality-assessment/em-fixed-points.txt -out /Users/tischer/Desktop/9.9.9/quality-assessment -tp /Users/tischer/Documents/detlev-arendt-clem-registration/transformations-new/TransformParameters.BSpline10-3Channels.0.txt


file <- "/Users/tischer/Desktop/9.9.9/quality-assessment/outputpoints.txt"
data <- read.table(file, sep = "" , header = F , nrows = 100,
                   na.strings ="", stringsAsFactors= F)

# transformed ( in mm, thus * 1000 to convert to um )
tx <- data[31] * 1000
ty <- data[32] * 1000
tz <- data[33] * 1000

# manual
library("readxl")

file <- "/Users/tischer/Desktop/9.9.9/quality-assessment/manually_matched_points.xlsx"
data <- read_excel( file )

mx = data[1] * 0.55 / 0.5
my = data[2] * 0.55 / 0.5
mz = data[3] * 0.55 / 0.5

distance <- sqrt( ( tx - mx )^2 + ( ty - my )^2 + ( tz - mz )^2 )
print( paste( "mean distance [um]:", mean( distance ) ) )
print( paste( "median distance [um]:", median( distance ) ) )

