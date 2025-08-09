import os
import json
import shutil

'''This script will get a subfolder list json file and copy all the images in the subfolder to a new output folder'''

def copy_images_to_output(image_list_file, root_folder, output_folder):
    # Load the list of image filenames from the JSON file
    with open(image_list_file, "r") as file:
        image_list = set(json.load(file)["images"])

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the image list and copy images to the output folder
    for image_entry in image_list:
        subfolder, filename = os.path.split(image_entry)
        source_path = os.path.join(root_folder, subfolder, filename)
        destination_path = os.path.join(output_folder, filename)
        
        shutil.copy2(source_path, destination_path)
        print(f"Copied {filename} to {output_folder}")

# Example usage:
image_list_file = "test_data/image_list_t_20.json"
root_folder = "test_data"
output_folder = "output"

copy_images_to_output(image_list_file, root_folder, output_folder)