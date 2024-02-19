import os
import re
import shutil


def copy_images(source_folder, destination_folder):
    """Copy files to processed

    Args:
        source_folder (str): folder where are stored source images
        destination_folder (str): folder where are copied images
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Traverse through the subdirectories in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file is an image
            if file.endswith(".jpg") or file.endswith(".png"):
                # Build the full paths of source and destination files
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                # Copy the file to the destination folder
                shutil.copy(source_path, destination_path)


def keep_only_masks(root_folder):
    """Filter files with 'labelIds in their name'

    Args:
        root_folder (str): Folder where are stored sub folder
    """
    # Regular expression pattern to search for 'labelIds' in the file name
    pattern = re.compile(r'labelIds')

    # Traverse through the subfolders of the root folder
    for subdir, _, files in os.walk(root_folder):
        # Check if the folder contains 'masks' in its name
        if 'masks' in subdir:
            # Iterate over the files in the folder
            for file in files:
                # Check if the file name contains 'labelIds'
                if re.search(pattern, file):
                    # Full path of the file
                    file_path = os.path.join(subdir, file)
                    # print("Keep:", file_path)
                else:
                    # Remove files that do not contain 'labelIds' in their name
                    file_path = os.path.join(subdir, file)
                    os.remove(file_path)
                    # print("Remove:", file_path)


def rename_files(root_folder):
    """Rename file names so that images and masks have the same name

    Args:
        root_folder (stg): Dataset where are stored all sub folder train, test and val
    """
    # Regular expression pattern to extract the required parts of the file name
    pattern = re.compile(r'(\w+)_(\d+)_(\d+)_\w+\.png')

    # Traverse through the subfolders of the root folder
    for subdir, _, files in os.walk(root_folder):
        # Iterate over the files in the folder
        for file in files:
            # Match the pattern with the file name
            match = re.match(pattern, file)
            if match:
                # Extract required parts from the file name
                prefix = match.group(1)
                middle = match.group(2)
                suffix = match.group(3)
                # Construct the new file name
                new_name = f"{prefix}_{middle.zfill(5)}_{suffix.zfill(6)}.png"
                # Full path of the file
                old_path = os.path.join(subdir, file)
                new_path = os.path.join(subdir, new_name)
                # Rename the file
                os.rename(old_path, new_path)
                # print(f"Renamed {old_path} to {new_path}")
