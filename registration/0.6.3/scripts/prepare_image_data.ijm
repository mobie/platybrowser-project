/*
 * Prepare ProSPr and corresponding EM tissue segmentations 
 * for registration with elastix.
 * 
 * 
 * Author: Christian Tischer
 */


prosprVoxelSize = 0.55; // micrometer
suffix = "-" + prosprVoxelSize + "um.tif";
version = "n.n.n";

names = newArray(1);
names[ 0 ] = "nuclei";

//
// Prepare ProSPr data
//
run("Close All");

inputDir = "/Users/tischer/Desktop/"+version+"/ProSPr";

files = newArray(1);
files[ 0 ] = "nuclei.tif";

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

files = newArray(1);
files[ 0 ] = "nuclei.tif";

inputDir = "/Users/tischer/Desktop/"+version+"/EM";

mergeString = ""
for (i = 0; i < names.length; i++) 
{
	open( inputDir + "/" + files[ i ] ); rename( names[ i ] );
	binarise( names[ i ] );
	scale( names[ i ] );
	c = i + 1; mergeString += "c" + c + "=" + names[ i ] + " "; 
}

run("Merge Channels...", mergeString+"create ignore keep");
rename("EM");
save( inputDir + "/EM" + suffix );

// 
// Prepare EM mask
// - dilate nuclei isotropically
//

if( names.length == 1 )
	selectWindow("EM"); rename( names[0] );

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
	run("Gaussian Blur 3D...", "x=4 y=4 z=4");
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

