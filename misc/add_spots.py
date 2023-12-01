import mobie.metadata

source_name = "nuclei_centers"
view = mobie.metadata.get_default_view(source_type="spots", source_name="nuclei_centers", menu_name="spots", )

print("Adding spots...")

mobie.add_spots(
    input_table="/Users/tischer/Documents/platybrowser-project/misc/centroids.tsv",
    root="/Users/tischer/Documents/platybrowser-project/data",
    dataset_name="1.0.1",
    spot_name=source_name,
    menu_name="spots",
    unit="micrometer",
    view=view,
)

print("done!")
