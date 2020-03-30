# Correcting the cell segmentation


## Step 0: Set-up

- We provide Horizon VMs at EMBL for proof-reading. See [the EMBL intranet](https://intranet.embl.de/it_services/services/horizon/index.html) for details on the EMBL VM set-up. For now, you can use the following VMs for proof-reading (you need to get access granted by IT first):
  - `VM Kreshuk 10`
- The segmentation has been divided into 50 sub-projects for proof-reading. As a proof-reader, you should always work on one project at a time. When you start working on a new project, please check [this spreadsheat](https://docs.google.com/spreadsheets/d/1w3w4ThrVEm4pCAjQ6oTeiNP91J-EHbXLbVxVHncBAo0/edit#gid=0), choose a block that has not been taken yet (`Proofreader` is blank) and put your name into the `Proofreader` column.

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
- From the field `Dataset` select `volumes/paintera` and press `Ok`. This will open the segmentation layer.
- Now you are almost good to go; you only need to scroll through the dataset until you find the part to proofread in this project (all the rest is greyed out!).

If you continue working on a project you have opened before, paintera will automatically start up in your last position.

In the first round of proof-reading, please focus on the following corrections / annotations:
- Correct cells which are fragmented into several parts or can be corrected by minor painting.
- Mark cells as correct that don't contain errors (including the ones you have corrected). This is done by pressing `l`, which will also hide the cell.
- Flag objects with errors you cannot correct right now: objects that contain parts of several cells incorrectly merged together and objects that are fragmented, but for which not all parts were assigned to your project (i.e. other parts of this cell are greyed out). This is done by pressing `e`, which will also hide the object. We will correct these objects later on.

Once all the cells assigned to your project are hidden because you have marked them as correct or flagged them as containing an error,
move to [Step2](https://github.com/platybrowser/platybrowser/tree/more-validation/segmentation/correction#step-2-split-falsely-merged-objects-with-seeds).
For details on paintera usage, see [the paintera Readme](https://github.com/saalfeldlab/paintera#usage).

**IMPORTANT:** when you quit paintera, either to take a break or to move on to [Step2](https://github.com/platybrowser/platybrowser/tree/more-validation/segmentation/correction#step-2-split-falsely-merged-objects-with-seeds), always safe your project.


## Step 2: Split falsely merged objects with seeds

TODO


## Step 3: Proof-reading with paintera

TODO
