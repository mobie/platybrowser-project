run("Close All");

inputDir = "/Users/tischer/Desktop/9.9.9";
runSimilarity = false;
runBSpline100 = false;
runBSpline30 = false;
runBSpline10 = true;
runTransformix = false;

if ( runSimilarity )
{
	run("Elastix", "elastixdirectory=/Applications/elastix_macosx64_v4.8 workingdirectory=/Users/tischer/Desktop/elastix-tmp fixedimagefile="+inputDir+"/EM/EM-0.55um.tif movingimagefile="+inputDir+"/ProSPr/ProSPR-0.55um.tif transformationtype=Similarity bsplinegridspacing=600,600 numiterations=3000 numspatialsamples=10000 gaussiansmoothingsigmas=15,15,15 transformationoutputfile="+inputDir+"/transformations/TransformParameters.Similarity.9.9.9.txt outputmodality=[Show output in ImageJ1] usefixedmask=true fixedmaskfile="+inputDir+"/EM/EM-mask-0.55um.tif usemovingmask=false movingmaskfile=/Users/tischer/Desktop/yeast_nobeads/fluo-pre-aligned-mask.tif useinitialtransformation=true initialtransformationfile="+inputDir+"/transformations/TransformParameters.Similarity.0.0.0.txt elastixparameters=Default finalresampler=FinalLinearInterpolator multichannelweights=1.0,1.0,1.0,1.0,1.0");
	mergeOutputChannels();
}

if ( runBSpline100 )
{
	run("Elastix", "elastixdirectory=/Applications/elastix_macosx64_v4.8 workingdirectory=/Users/tischer/Desktop/elastix-tmp fixedimagefile="+inputDir+"/EM/EM-0.55um.tif movingimagefile="+inputDir+"/ProSPr/ProSPR-0.55um.tif transformationtype=BSpline bsplinegridspacing=100,100,100 numiterations=3000 numspatialsamples=10000 gaussiansmoothingsigmas=10,10,10 transformationoutputfile="+inputDir+"/transformations/TransformParameters.BSpline100.9.9.9.txt outputmodality=[Show output in ImageJ1] usefixedmask=true fixedmaskfile="+inputDir+"/EM/EM-mask-0.55um.tif usemovingmask=false movingmaskfile=/Users/tischer/Desktop/yeast_nobeads/fluo-pre-aligned-mask.tif useinitialtransformation=true initialtransformationfile="+inputDir+"/transformations/TransformParameters.Similarity.9.9.9.txt elastixparameters=Default finalresampler=FinalLinearInterpolator multichannelweights=1.0,1.0,1.0,1.0,1.0");
	mergeOutputChannels();
}

if ( runBSpline30 )
{
	run("Elastix", "elastixdirectory=/Applications/elastix_macosx64_v4.8 workingdirectory=/Users/tischer/Desktop/elastix-tmp fixedimagefile="+inputDir+"/EM/EM-0.55um.tif movingimagefile="+inputDir+"/ProSPr/ProSPR-0.55um.tif transformationtype=BSpline bsplinegridspacing=30,30,30 numiterations=3000 numspatialsamples=10000 gaussiansmoothingsigmas=7,7,7 transformationoutputfile="+inputDir+"/transformations/TransformParameters.BSpline30.9.9.9.txt outputmodality=[Show output in ImageJ1] usefixedmask=true fixedmaskfile="+inputDir+"/EM/EM-mask-0.55um.tif usemovingmask=false movingmaskfile=/Users/tischer/Desktop/yeast_nobeads/fluo-pre-aligned-mask.tif useinitialtransformation=true initialtransformationfile="+inputDir+"/transformations/TransformParameters.BSpline100.9.9.9.txt elastixparameters=Default finalresampler=FinalLinearInterpolator multichannelweights=1.0,1.0,1.0,1.0,1.0");
	mergeOutputChannels();
}

if ( runBSpline10 )
{
	run("Elastix", "elastixdirectory=/Applications/elastix_macosx64_v4.8 workingdirectory=/Users/tischer/Desktop/elastix-tmp fixedimagefile="+inputDir+"/EM/EM-0.55um.tif movingimagefile="+inputDir+"/ProSPr/ProSPR-0.55um.tif transformationtype=BSpline bsplinegridspacing=10,10,10 numiterations=3000 numspatialsamples=10000 gaussiansmoothingsigmas=0,0,0 transformationoutputfile="+inputDir+"/transformations/TransformParameters.BSpline10.9.9.9.txt outputmodality=[Show output in ImageJ1] usefixedmask=true fixedmaskfile="+inputDir+"/EM/EM-mask-0.55um.tif usemovingmask=false movingmaskfile=/Users/tischer/Desktop/yeast_nobeads/fluo-pre-aligned-mask.tif useinitialtransformation=true initialtransformationfile="+inputDir+"/transformations/TransformParameters.BSpline30.9.9.9.txt elastixparameters=Default finalresampler=FinalLinearInterpolator multichannelweights=1.0,1.0,1.0,1.0,1.0");
	mergeOutputChannels();
}

if ( runTransformix )
{
	run("Transformix", "elastixdirectory=/Applications/elastix_macosx64_v4.8 workingdirectory=/Users/tischer/Desktop/elastix-tmp inputimagefile=/Users/tischer/Desktop/9.9.9/ProSPr/ProSPr-0.55um-muscles-neuropil-nuclei-gut.zip transformationfile=/Users/tischer/Desktop/9.9.9/transformations/TransformParameters.BSpline10.9.9.9.txt outputmodality=[Show images] outputfile=/Users/tischer/Desktop/stomach-prospr-registered numthreads=4");

	mergeString = "";
	for (i = 0; i < 4; i++) {
		c = i + 1;
		mergeString += "c"+c+"=transformed-ch"+i+" ";
	}
	run("Merge Channels...", mergeString + "create ignore");
	rename("transformed-moving");
}


run("Synchronize Windows");


function mergeOutputChannels()
{
	selectWindow("fixed");
	getDimensions(w, h, channels, slices, frames);
	if ( channels == 1 ) return;
	mergeString = "";
	for (i = 0; i < channels; i++) {
		c = i + 1;
		mergeString += "c"+c+"=transformed-ch"+i+" ";
	}
	run("Merge Channels...", mergeString + "create ignore");
	rename("transformed-moving");
}
