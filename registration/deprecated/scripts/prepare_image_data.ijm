/*
 * Prepare ProSPr and corresponding EM tissue segmentations 
 * for registration with elastix.
 * 
 * 
 * Author: Christian Tischer
 */


prosprVoxelSize = 0.55; // micrometer
suffix = "-" + prosprVoxelSize + "um.tif";

names = newArray(4);
names[ 0 ] = "muscles";
names[ 1 ] = "neuropil";
names[ 2 ] = "nuclei";
names[ 3 ] = "stomach";

//
// Prepare ProSPr data
//
run("Close All");

inputDir = "/Users/tischer/Desktop/9.9.9/ProSPr";

files = newArray(4);
files[ 0 ] = "muscles.tif";
files[ 1 ] = "neuropil.tif"; 
files[ 2 ] = "nuclei.tif";
files[ 3 ] = "stomach.tif"; 

mergeString = ""
for (i = 0; i < names.length; i++) 
{
	open( inputDir + "/" + files[ i ] ); rename( names[ i ] );
	c = i + 1; mergeString += "c" + c + "=" + names[ i ] + " "; 
}
run("Merge Channels...", mergeString+"create ignore keep");
rename("ProSPr");
save( inputDir + "/ProSPr" + suffix );

//
// Prepare EM segmentations
//
run("Close All");

files = newArray(4);
files[ 0 ] = "em-segmented-muscles-ariadne.tif";
files[ 1 ] = "em-segmented-neuropil-ariadne.tif"; 
files[ 2 ] = "sbem-6dpf-1-whole-segmented-nuclei.tif";
files[ 3 ] = "em-segmented-stomach.tif"; 

inputDir = "/Users/tischer/Desktop/9.9.9/EM";

mergeString = ""
for (i = 0; i < names.length; i++) 
{
	open( inputDir + "/" + files[ i ] ); rename( names[ i ] );
	scale( names[ i ] );
	binarise( names[ i ] );
	c = i + 1; mergeString += "c" + c + "=" + names[ i ] + " "; 
}

run("Merge Channels...", mergeString+"create ignore keep");
rename("EM");
save( inputDir + "/EM" + suffix );

// 
// Prepare EM mask
//

for (i = 0; i < names.length; i++) 
	dilateBinaryFast( names[ i ] ); 

run("Merge Channels...", mergeString+"create ignore keep");
rename("EM-Mask");
save( inputDir + "/EM-mask" + suffix );

//
// Functions
//

function dilateBinaryFast( name )
{
	selectWindow( name );
	run("Gaussian Blur 3D...", "x=10 y=10 z=10");
	setThreshold(1, 255);
	setOption("BlackBackground", true);
	run("Convert to Mask", "method=Default background=Dark black");
	rename( name );
}

function binarise( name )
{
	selectWindow( name );
	setThreshold(1, 65535);
	setOption("BlackBackground", true);
	run("Convert to Mask", "method=Default background=Dark black");
	rename( name );
}

function openScaled( image )
{
	open( inputDir + "/" + image + suffix );
}

function scale( name ) {
	selectWindow( name );
	
	//getVoxelSize(width, height, depth, unit);
	//scaleX = 1.0 * width / prosprVoxelSize;
	//scaleY = 1.0 * height / prosprVoxelSize;
	//scaleZ = 1.0 * depth / prosprVoxelSize;
	//run("Scale...", "x=&scaleX y=&scaleY z=&scaleZ interpolation=Bilinear average process create");
	
	// below command is more practical (as compared to above), because
	// it is ensured that all images have the exact same number of voxels
	run("Scale...", "width=500 height=471 depth=519 interpolation=Bilinear average process create");
	rename( "scaled" );
	selectWindow( name ); close( );
	selectWindow( "scaled" );
	rename( name );
}

