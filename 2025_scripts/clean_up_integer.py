#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Union

"""
Clean up the integer formatting of the tables
"""

FOLDER_PATH: Path = Path('/home/cros/bioinformatics/platybrowser-project-2025/data/platybrowser_6dpf/tables') 


def is_integer_like_float(value: str) -> bool:
    """
    Checks if a string represents an integer formatted as a float (e.g., "123.0").
    Returns True only if the string ends with ".0" and the part before it is an integer.
    """
    if not isinstance(value, str) or not value.endswith('.0'):
        return False
    try:
        # Check if the part before ".0" can be converted to an integer.
        int(value[:-2])
        return True
    except (ValueError, TypeError):
        return False


def process_tsv_file(file_path: Path):
    """
    Reads a TSV file, cleans integer-like float columns, and overwrites the file.
    """
    print(f"Processing '{file_path.name}'...")

    try:
        # --- 1. Read the entire file into memory ---
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            data = list(reader)

        if not data or len(data) < 2:
            print("  - Skipping: File is empty or has no data rows.")
            return

        header = data[0]
        num_columns = len(header)
        data_rows = data[1:]
        
        cols_to_clean = []
        for col_index in range(num_columns):
            is_cleanable = True
            for row in data_rows:
                if col_index < len(row) and row[col_index]:
                    if not is_integer_like_float(row[col_index]):
                        is_cleanable = False
                        break
            
            if is_cleanable:
                cols_to_clean.append(col_index)

        if not cols_to_clean:
            print("  - No columns to clean.")
            return

        print(f"  - Found {len(cols_to_clean)} columns to clean: {cols_to_clean}")


        for row in data_rows:
            for col_index in cols_to_clean:
                if col_index < len(row) and is_integer_like_float(row[col_index]):
                    row[col_index] = row[col_index][:-2]

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(header)
            writer.writerows(data_rows)
            
        print(f"  - Successfully cleaned and saved file.")

    except Exception as e:
        print(f"  - ERROR processing {file_path.name}: {e}")


def main():
    """
    Main function to find and process all TSV files in the configured folder.
    """
    print(f"--- Starting TSV cleanup in folder: '{FOLDER_PATH.resolve()}' ---")
    
    found_files = False
    for file_path in FOLDER_PATH.rglob('*.tsv'):
        found_files = True
        process_tsv_file(file_path)
            
    if not found_files:
        print("No .tsv files found in the specified folder.")
    
    print("--- Cleanup complete. ---")


if __name__ == '__main__':
    main()