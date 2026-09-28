#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "boto3",
#   "python-dotenv" 
# ]
# ///

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from dotenv import load_dotenv

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    print("Boto3 library not found. Please install it to use S3 functionality.")
    print("Run: pip install boto3")
    # We don't exit here, to allow XML generation in dry-run mode without boto3.
    boto3 = None

def upload_directory_to_s3(local_path: Path, bucket_name: str, s3_prefix: str, s3_client):
    """
    Recursively uploads a directory to an S3 bucket.

    Args:
        local_path (Path): The local directory to upload.
        bucket_name (str): The name of the S3 bucket.
        s3_prefix (str): The prefix to use for the S3 object key.
        s3_client: The initialized boto3 S3 client.
    """
    if not boto3:
        print("Error: Boto3 is not installed. Cannot perform S3 upload.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Uploading directory '{local_path}' to bucket '{bucket_name}'...")
    
    try:
        for root, _, files in os.walk(local_path):
            for filename in files:
                local_file = Path(root) / filename
                # Create a relative path to maintain directory structure in S3
                relative_path = local_file.relative_to(local_path.parent)
                s3_key = f"{s3_prefix}/{relative_path.as_posix()}"

                print(f"  Uploading {local_file} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(str(local_file), bucket_name, s3_key)

        print("Upload completed successfully.")
    except NoCredentialsError:
        print("Error: S3 credentials not found. Please configure them.", file=sys.stderr)
        sys.exit(1)
    except ClientError as e:
        print(f"Error: A client error occurred during S3 upload: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during upload: {e}", file=sys.stderr)
        sys.exit(1)


def process_xml_file(xml_path: Path, args: argparse.Namespace):
    """
    Processes a single XML file: uploads its .n5 folder and creates a new XML.
    
    Args:
        xml_path (Path): The path to the input XML file.
        args (argparse.Namespace): The command-line arguments.
    """
    print(f"\n--- Processing {xml_path.name} ---")
    
    # 1. Check for the corresponding .n5 folder
    n5_folder_name = xml_path.stem + ".n5"
    n5_path = xml_path.parent / n5_folder_name
    
    if not n5_path.is_dir():
        print(f"Warning: Corresponding .n5 folder '{n5_path}' not found. Skipping.")
        return

    # 2. Parse the XML file
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error: Could not parse XML file '{xml_path}'. Skipping. Details: {e}", file=sys.stderr)
        return

    # 3. Find and modify the ImageLoader section
    image_loader = root.find(".//ImageLoader")
    if image_loader is None:
        print(f"Warning: <ImageLoader> tag not found in '{xml_path}'. Skipping.")
        return
        
    # === NEW CHANGE: Update the format attribute to bdv.n5.s3 ===
    image_loader.set('format', 'bdv.n5.s3')
    # ==========================================================

    n5_tag = image_loader.find("n5")
    if n5_tag is None:
        print(f"Warning: Local <n5> tag not found in '{xml_path}'. Assuming it's already processed. Skipping.")
        return

    # Remove the old local n5 tag
    image_loader.remove(n5_tag)
    
    # Add the new S3 tags
    # Ensure the prefix ends with a slash if it's not empty
    s3_key_prefix = args.s3_prefix
    if s3_key_prefix and not s3_key_prefix.endswith('/'):
        s3_key_prefix += '/'

    ET.SubElement(image_loader, "Key").text = f"{s3_key_prefix}{n5_folder_name}"
    ET.SubElement(image_loader, "SigningRegion").text = args.signing_region
    ET.SubElement(image_loader, "ServiceEndpoint").text = args.service_endpoint
    ET.SubElement(image_loader, "BucketName").text = args.bucket_name
    
    # This helps format the output XML nicely. Requires Python 3.9+
    if sys.version_info >= (3, 9):
        ET.indent(tree, space="  ")

    # 4. Write the new XML file to the output directory
    output_xml_path = args.output_folder / xml_path.name
    print(f"Generating new XML file at: {output_xml_path}")
    
    # Write the XML without the declaration
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=False)
    
    # Append an extra newline at the end of the file
    with open(output_xml_path, "a", encoding="utf-8") as f:
        f.write("\n")
    # 5. Handle S3 Upload if not a dry run
    if not args.dry_run:
        s3_client = boto3.client(
            's3',
            endpoint_url=args.service_endpoint,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            region_name=args.signing_region
        )
        
        s3_n5_folder_prefix = f"{s3_key_prefix}{n5_folder_name}/"
        
        # Only check for existence if we are NOT forcing the upload
        if not args.force_upload:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=args.bucket_name,
                    Prefix=s3_n5_folder_prefix,
                    MaxKeys=1
                )
                if response.get('Contents'):
                    print(f"S3 folder s3://{args.bucket_name}/{s3_n5_folder_prefix} already exists. Skipping upload.")
                    return  # Skip to the next XML file
            except ClientError as e:
                print(f"Error checking S3 prefix s3://{args.bucket_name}/{s3_n5_folder_prefix}: {e}", file=sys.stderr)
                print("Cannot proceed with upload. Exiting.")
                sys.exit(1)
        else:
            print(f"Force upload enabled. Overwriting s3://{args.bucket_name}/{s3_n5_folder_prefix} if it exists...")

        # Proceed with upload (either it didn't exist, or force_upload is True)
        upload_directory_to_s3(n5_path, args.bucket_name, s3_key_prefix.rstrip('/'), s3_client)
    else:
        print(f"DRY RUN: Skipping upload of '{n5_path}'.")

def main():
    """Main function to parse arguments and start the processing."""
    # For S3 secrets
    load_dotenv()

    parser = argparse.ArgumentParser(
    )
    
    parser.add_argument("-i", "--input-folder", type=Path, required=True, help="Path to the input folder containing source XML and .n5 files.")
    parser.add_argument("-o", "--output-folder", type=Path, required=True, help="Path to the folder where new XML files will be created.")
    parser.add_argument("-b", "--bucket-name", required=True, help="S3 bucket name.")
    parser.add_argument("-e", "--service-endpoint", required=True, help="S3 service endpoint URL (e.g., https://s3.your-minio.com).")
    parser.add_argument("-r", "--signing-region", required=True, help="S3 signing region (e.g., us-east-1).")
    parser.add_argument("-p", "--s3-prefix", default="", help="A common path prefix for the S3 object key (e.g., '0.6.3/images/local/').")
    parser.add_argument("--aws-access-key-id", help="S3 Access Key ID. Defaults to AWS_ACCESS_KEY_ID env var.")
    parser.add_argument("--aws-secret-access-key", help="S3 Secret Access Key. Defaults to AWS_SECRET_ACCESS_KEY env var.")
    parser.add_argument("--dry-run", action="store_true", help="Run the script without uploading to S3. It will only generate the new XML files.")
    parser.add_argument("--force-upload", action="store_true", help="Force upload to S3 even if the destination folder already exists.")
    
    args = parser.parse_args()
    
    args = parser.parse_args()

    # We read from the env
    access_key = args.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = args.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    if not args.dry_run and (not access_key or not secret_key):
        print("Error: AWS credentials not found.", file=sys.stderr)
        print("Please provide them via command-line arguments or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.", file=sys.stderr)
        sys.exit(1)

    # Store the final credentials back into args so process_xml_file can find them
    args.aws_access_key_id = access_key
    args.aws_secret_access_key = secret_key
    
    # Validate paths
    if not args.input_folder.is_dir():
        print(f"Error: Input folder '{args.input_folder}' not found.", file=sys.stderr)
        sys.exit(1)
        
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("**** RUNNING IN DRY-RUN MODE. NO FILES WILL BE UPLOADED. ****")

    # Iterate through XML files in the input folder
    xml_files_found = 0
    for file_path in args.input_folder.glob("*.xml"):
        process_xml_file(file_path, args)
        xml_files_found += 1
        
    if not xml_files_found:
        print("No .xml files found in the input directory.")
    else:
        print("\n--- All files processed. ---")

if __name__ == "__main__":
    main()