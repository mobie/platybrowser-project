#!/usr/bin/env -S uv run

import json
import csv
import os
import argparse
from pathlib import Path

def create_new_source(folder_name, xml_file_path):
    """
    Creates a new 'source' dictionary for the dataset.json.

    Args:
        folder_name (str): The name of the folder containing the XML, e.g., 'vergara_2021'.
        xml_file_path (str): The full name of the XML file, e.g., 'prospr-6dpf-1-whole-allcr2.xml'.

    Returns:
        dict: The structured dictionary for the new source.
    """
    relative_path = f"images/bdv-n5-s3/{folder_name}/{xml_file_path}"
    return {
        "image": {
            "imageData": {
                "bdv.n5.s3": {
                    "relativePath": relative_path
                }
            }
        }
    }

def create_new_view(source_name, ui_group):
    """
    Creates a new 'view' dictionary for the dataset.json.

    Args:
        source_name (str): The name for the new source (e.g., 'allcr2').
        ui_group (str): The name of the uiSelectionGroup for this view.

    Returns:
        dict: The structured dictionary for the new view.
    """
    return {
        "uiSelectionGroup": ui_group,
        "sourceDisplays": [
            {
                "imageDisplay": {
                    "sources": [source_name],
                    "color": "randomFromGlasbey",
                    "contrastLimits": [0.0, 1.0],
                    "showImagesIn3d": False,
                    "invert": False,
                    "name": source_name,
                    "opacity": 1.0,
                    "visible": True
                }
            }
        ],
        "isExclusive": False
    }

def update_dataset_json(json_path, image_folder, csv_path, ui_group):
    """
    Updates the dataset.json file with new sources and views.

    Args:
        json_path (str): The file path for the dataset.json file.
        image_folder (str): The folder containing the new XML files (e.g., 'vergara_2021').
        csv_path (str): The file path for the CSV mapping file.
        ui_group (str): The name for the new uiSelectionGroup.
    """
    try:
        # Load the existing dataset
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded {json_path}")

        # Read the CSV mappings
        mappings = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                mappings[row[0]] = row[1]
            print(f"Loaded {len(mappings)} mappings from {csv_path}")

        # Prepare to add new entries
        folder_name = Path(image_folder).name
        added_count = 0
        
        # Check if sources and views keys exist, if not, create them
        if 'sources' not in data:
            data['sources'] = {}
        if 'views' not in data:
            data['views'] = {}

        # Iterate through the files in the specified folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".xml") and filename in mappings:
                short_name = mappings[filename]
                
                # Check for duplicates before adding
                if short_name in data['sources']:
                    print(f"Source '{short_name}' already exists. Skipping.")
                    continue
                
                # Create and add the new source and view
                print(f"Processing: {filename} -> {short_name}")
                new_source = create_new_source(folder_name, filename)
                new_view = create_new_view(short_name, ui_group)
                
                data['sources'][short_name] = new_source
                data['views'][short_name] = new_view
                added_count += 1

        # Write the updated data back to the file
        if added_count > 0:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\nSuccessfully added {added_count} new entries to {json_path}")
        else:
            print("\nNo new entries were added.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update dataset.json with new image sources and views.")
    parser.add_argument("json_file", help="Path to the dataset.json file.")
    parser.add_argument("image_folder", help="Path to the folder containing XML files (e.g., 'vergara_2021').")
    parser.add_argument("csv_file", help="Path to the CSV file mapping XML filenames to short names.")
    parser.add_argument("ui_group", help="Name for the new uiSelectionGroup (e.g., 'stainings-2025-paper').")

    args = parser.parse_args()

    update_dataset_json(args.json_file, args.image_folder, args.csv_file, args.ui_group)