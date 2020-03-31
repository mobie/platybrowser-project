# Correcting the cell segmentation


## Step 0: Set-up

- We provide Horizon VMs at EMBL for proof-reading. See [the EMBL intranet](https://intranet.embl.de/it_services/services/horizon/index.html) for details on the EMBL VM set-up. For now, you can use the following VMs for proof-reading (you need to get access granted by IT first):
  - `VM Kreshuk 10`
- The segmentation has been divided into 50 sub-projects for proof-reading. As a proof-reader, you should always work on one project at a time. When you start working on a new project, please check [this spreadsheat](https://docs.google.com/spreadsheets/d/1w3w4ThrVEm4pCAjQ6oTeiNP91J-EHbXLbVxVHncBAo0/edit#gid=0), choose a block that has not been taken yet (`Proofreader` is blank) and put your name into the `Proofreader` column.
- In order to estimate the total time spent on proof-reading, please keep track of the time you spent on the individual steps and mark them in the corresponding columns as HH:MM.

Once you are logged into your VM, open a terminal by clicking `Applications -> System Tools -> Terminal`.
In the terminal, enter
```sh
cd /g/arendt/EM_6dpf_segmentation/corrections_and_proofreading
```

## Step 1: Proof-reading with paintera

Start paintera for your project via
```sh
./run_paintera.sh paintera_projects/project<ID>
```
where `<ID>` is the id of your project, e.g. `01` if you are working on project number 1 or `10` if you are working on project 10.

This will open am empty paintera window. Now, you need to open your project in paintera:
- Press `Ctrl + o` select `N5`, which will open the paintera file menu.
- In the field next to `Find`, type in the path to your project's data: `/g/arendt/EM_6dpf_segmentation/corrections_and_proofreading/paintera_projects/project<ID>/data.n5`
- From the field `Dataset` select `volumes/raw` and press `Ok`. This will open the raw data. 
- Now you also need to load the segmentation. Press `Ctrl + o` and select `N5` again.
- Switch the field `Type` from `RAW` to `LABEL`. From the field `Dataset` select `volumes/paintera` and press `Ok`. This will open the segmentation layer.
- Now you are almost good to go; you only need to scroll through the dataset until you find the part to proofread in this project (all the rest is greyed out!).

If you continue working on a project you have opened before, paintera will automatically start up in your last position.

In the first round of proof-reading, please focus on the following corrections / annotations:
- Correct cells which are fragmented into several parts or can be corrected by minor painting.
- Mark cells as correct that don't contain errors (including the ones you have corrected). This is done by pressing `l`, which will also hide the cell.
- Flag objects with errors you cannot correct right now: objects that contain parts of several cells incorrectly merged together and objects that are fragmented, but for which not all parts were assigned to your project (i.e. other parts of this cell are greyed out). This is done by pressing `e`, which will also hide the object. We will correct these objects later on.

Once all the cells assigned to your project are hidden because you have marked them as correct or flagged them as containing an error,
move to [Step2](https://github.com/platybrowser/platybrowser/tree/more-validation/segmentation/correction#step-2-split-falsely-merged-objects-with-seeds).
For details on paintera usage, see [the paintera Readme](https://github.com/saalfeldlab/paintera#usage).

**IMPORTANT:** when you quit paintera, either to take a break or to move on to [Step2](https://github.com/platybrowser/platybrowser/tree/more-validation/segmentation/correction#step-2-split-falsely-merged-objects-with-seeds), always safe your project by pressing `Save and quit` after pressing the `x` button. If another window pops up after this, press `Commit`.
 

## Step 2: Split falsely merged objects with seeds

The `splitting tool` allows to split objects by providing seeds for the different cells, cell fragments or extracellular tissue parts
erronously merged together. It is implemented using [napari](https://github.com/napari/napari#napari). Start it via
```sh
./run_splitter.sh paintera_projects/project<ID>
```
If you start it the first time for this block, this will need to run some pre-processing that can take up to a few hours.
After this has run through, run the command again. It will open up a window showing one of the objects to be corrected:

![Split1](https://github.com/platybrowser/platybrowser/blob/master/segmentation/correction/ims/split1.png)

You can move around the viewer like this:
- Drag the slider (below the image) to scroll through the z-axis.
- Use the scroll wheel to zoom in/out.
- Hold and drag the left or right mouse button to pan the view.

The viewer contains four different layers:
- `raw`: the raw data
- `ws`: the small fragments making up the object
- `seg`: the mask for the current object
- `seeds`: the seeds you provide to split the object
These layers can be toggled by clicking the eye symbol next to them.

Note that you do not have to split all objects that will be loaded by the correction tool, because you may have flagged some objects with `e` in [Step1](https://github.com/platybrowser/platybrowser/tree/master/segmentation/correction#step-1-proof-reading-with-paintera) for other reasons than false merges. In this case, skip the current object py pressing `s` and closing the current window (`x` symbol in the top right). The next object will be loaded automatically.

In order to split an object, you need to provide seeds for the different sub-parts that will be used to
grow the corresponding objects. For this, select the `seeds` layer by clicking on it; then enable the `paint mode` by clicking the brush symbol and increase the seed id by clicking the `+` symbol next to `label`:

![Split2](https://github.com/platybrowser/platybrowser/blob/master/segmentation/correction/ims/split2.png)

Now, paint with different seed ids for the parts of the object that shall be split.
In order to move around the volume easier you can switch back and forth between `view mode` / `paint mode` by clicking 
the magnifying glass symbol / the brush symbol.

![Split3](https://github.com/platybrowser/platybrowser/blob/master/segmentation/correction/ims/split3.png)

Once you have painted seeds for all parts, press `w`. This will grow the segmentation for all sub-parts:

![Split4](https://github.com/platybrowser/platybrowser/blob/master/segmentation/correction/ims/split4.png)

Note: sometimes the tool might not react to pressing keys; in this case, switch back to the `view mode` (magnifying glass) and pan the view (hold left click somewhere in the image and drag the mouse). Now it should react to pressing keys again.
Continue painting seeds and updating the segmentation until you are satisfied with the resulting segmentations.
Note that it is often not possible to get perfect results at boundaries because the splitting workflow cannot correct issues 
in the underlying fragments (which you can see by activating the `ws` layer).
Once you are done, just close the window (`x` symbol in the top right); the tool will open the window for the next object until all objects have been corrected or skipped.

If you want to take a break, close the tool by pressing `q`. Your progress will be saved.


## Step 3: Proof-reading with paintera

After you are finihed with [Step 2](https://github.com/platybrowser/platybrowser/blob/master/segmentation/correction/README.md#step-2-split-falsely-merged-objects-with-seeds), open paintera for the project agai (see [Step 1](https://github.com/platybrowser/platybrowser/blob/master/segmentation/correction/README.md#step-1-proof-reading-with-paintera)). This time, please revisit the objects you have corrected with the splitting tool, correct small remaining issues via painting or merging and lock correct cells with `l`. If the objects are still not fully correctable, mark them as containing an error with `e` again.

Once you are done, change the value in the `Done` column of [the spreadsheat](https://docs.google.com/spreadsheets/d/1w3w4ThrVEm4pCAjQ6oTeiNP91J-EHbXLbVxVHncBAo0/edit#gid=0) to `TRUE`, enter all times and move to the next project.
