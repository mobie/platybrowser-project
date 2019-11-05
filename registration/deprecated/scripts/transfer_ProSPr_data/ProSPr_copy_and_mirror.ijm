// Copy files from ProSPr to the EM working directory
// Mirror in the x axis (because the SP8 does it)
// Add dimension information
// Threshold them (Not the reference)
// Manage names

// Hernando MV. Sep2019

setBatchMode(true);
// directory of the curated MEDs
meds_dir = "/g/arendt/PrImR/ProSPr6/4SPM_Binarization/CuratedMEDs_Good/";
// ProSPr reference
ref_file = "/g/arendt/PrImR/ProSPr6/ProSPr6_Ref.tif";
// segmented data
trackem_dir = "/g/arendt/PrImR/ProSPr6/TrackEM/";
// output directory
outdir = "/g/arendt/EM_6dpf_segmentation/platy-browser-data/data/rawdata/prospr/";

// read dictionary
dic_file = "/g/arendt/EM_6dpf_segmentation/GenerationOfVirtualCells/ProSPr_VirtualCells/helper_files/OlegGenesDiccionary.txt";
filestring=File.openAsString(dic_file);
rows=split(filestring, "\n");
badname=newArray(rows.length);
goodname=newArray(rows.length);
for(i=0; i<rows.length; i++){
	columns=split(rows[i],",");
	badname[i]=columns[0];
	goodname[i]=columns[1];
}

// get a list of the med files
meds_files = getFileList(meds_dir);
filenames = newArray();
for (i=0;i<meds_files.length;i++){
	subfiles = getFileList(meds_dir+meds_files[i]);
	for (j=0;j<subfiles.length;j++){
		if (endsWith(subfiles[j],"_MEDs.tif")){
			pieces = split(subfiles[j],"_");
			// avoid inserting Pty2
			if (pieces[0] != "Pty2"){
				filenames = Array.concat(filenames,meds_dir+meds_files[i]+"/"+subfiles[j]);
			}
		}
	}
}
// add the segmented data
trackem_files = getFileList(trackem_dir);
for (i=0;i<trackem_files.length;i++){
	subfiles = getFileList(trackem_dir+trackem_files[i]);
	for (j=0;j<subfiles.length;j++){
		if (endsWith(subfiles[j],"_MEDs.tif")){
			pieces = split(subfiles[j],"_");
			// avoid inserting MB
			if (pieces[0] != "MB"){
				filenames = Array.concat(filenames,trackem_dir+trackem_files[i]+"/"+subfiles[j]);
			}
		}
	}
}

for (i = 0; i < filenames.length; i++) {
	// open
	open(filenames[i]);
	f_name = getTitle();
	print(f_name);
	// flip
	run("Flip Horizontally", "stack");
	// change properties
	run("Properties...", "channels=1 slices=251 frames=1 unit=micrometers pixel_width=0.55 pixel_height=0.55 voxel_depth=0.55");
	// threshold
	setThreshold(1, 255); 
	setOption("BlackBackground", false);
	run("Convert to Mask", "stack");
	// save
	// parse title
	pieces = split(f_name, "_");
	// find index of name
	idx = getIndex(pieces[0], badname);
	print(idx);
	// get correspondent name
	if(idx<9000){
		genename = goodname[idx];
		print(genename);
		new_name = pieces[0] + "-" + genename + "--prosprspaceMEDs.tif";
	}
	else{
		new_name = pieces[0] + "--prosprspaceMEDs.tif";
	}
	// correct Hox5
	if(pieces[0]=="Hox5"){
		new_name = "Hox4--prosprspaceMEDs.tif";
	}
	saveAs("Tiff", outdir + new_name);
	close(new_name);
}

// copy the reference
open(ref_file);
print(getTitle());
// flip
run("Flip Horizontally", "stack");
// change properties
run("Properties...", "channels=1 slices=251 frames=1 unit=micrometers pixel_width=0.55 pixel_height=0.55 voxel_depth=0.55");
// save
new_name = "ProSPr6-Ref--prosprspaceMEDs.tif";
saveAs("Tiff", outdir + new_name);
close(new_name);

run("Close All");
print("DONE");
run("Quit");

function getIndex(word, array){
	for (i = 0; i < array.length; i++) {
		if (word == array[i]) {
			return(i);		
		}
	}
	return(10000); // could not figure out how to make NaN work...
}
