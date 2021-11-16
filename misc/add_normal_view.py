import os
import mobie
import json

ds_folder = "../data/1.0.1"
level_file = os.path.join(ds_folder, "misc", "leveling.json")
with open(level_file) as f:
    normal = json.load(f)["NormalVector"]
print(normal)
trafo = mobie.metadata.get_viewer_transform(normal_vector=normal)

view = {"viewerTransform": trafo, "isExclusive": False, "uiSelectionGroup": "anatomical-views"}
mobie.metadata.add_view_to_dataset(ds_folder, "coronal", view)
