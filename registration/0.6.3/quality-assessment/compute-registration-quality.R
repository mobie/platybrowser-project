# Commands to produce the output points on the terminal (examples)
# 1. change dir into the folder with the transformation files; this is important because there are relative paths within the transformation files that rely on this.
# cd  /Users/tischer/Desktop/0.6.3/transformations/ 
# 2. run transformix to compute the transformed points
# Rotation: /Applications/elastix_macosx64_v4.8/run_transformix.sh -def /Users/tischer/Desktop/0.6.3/quality-assessment/em-fixed-points.txt -out /Users/tischer/Desktop/0.6.3/quality-assessment -tp /Users/tischer/Desktop/0.6.3/transformations/TransformParameters.RotationPreAlign.0.0.0.txt
# Similarity: /Applications/elastix_macosx64_v4.8/run_transformix.sh -def /Users/tischer/Desktop/0.6.3/quality-assessment/em-fixed-points.txt -out /Users/tischer/Desktop/0.6.3/quality-assessment -tp /Users/tischer/Desktop/0.6.3/transformations/TransformParameters.Similarity.9.9.9.txt
# BSpline100: /Applications/elastix_macosx64_v4.8/run_transformix.sh -def /Users/tischer/Desktop/0.6.3/quality-assessment/em-fixed-points.txt -out /Users/tischer/Desktop/0.6.3/quality-assessment -tp /Users/tischer/Desktop/0.6.3/transformations/TransformParameters.BSpline100.9.9.9.txt
# BSpline30: /Applications/elastix_macosx64_v4.8/run_transformix.sh -def /Users/tischer/Desktop/0.6.3/quality-assessment/em-fixed-points.txt -out /Users/tischer/Desktop/0.6.3/quality-assessment -tp /Users/tischer/Desktop/0.6.3/transformations/TransformParameters.BSpline30.9.9.9.txt
# BSpline10: /Applications/elastix_macosx64_v4.8/run_transformix.sh -def /Users/tischer/Desktop/0.6.3/quality-assessment/em-fixed-points.txt -out /Users/tischer/Desktop/0.6.3/quality-assessment -tp /Users/tischer/Desktop/0.6.3/transformations/TransformParameters.BSpline10.9.9.9.txt
input_folder = "/Users/tischer/Desktop/0.6.3/quality-assessment"

file <- paste(input_folder,"outputpoints.txt",sep="/")
data <- read.table(file, sep = "" , header = F , nrows = 100,
                   na.strings ="", stringsAsFactors= F)

# transformed ProSPr ( in mm, thus * 1000 to convert to um )
tx <- data[31] * 1000
ty <- data[32] * 1000
tz <- data[33] * 1000

# manually assigned ProSPr
library("readxl")

file <- paste(input_folder,"manually_matched_points.xlsx",sep="/") 
data <- read_excel( file )

mx = data[1] 
my = data[2] 
mz = data[3]

distance <- sqrt( ( tx - mx )^2 + ( ty - my )^2 + ( tz - mz )^2 )
print( paste( "mean distance [um]:", mean( distance ) ) )
print( paste( "median distance [um]:", median( distance ) ) )

