import fiftyone as fo
import os

frigate_dir = "/Users/algore/GIT/fiftyone/frigate-dataset"

# Collect all image files from the directory
image_files = [
    os.path.join(frigate_dir, f)
    for f in os.listdir(frigate_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Create dataset from images
dataset = fo.Dataset.from_images(image_files, name="frigate2", overwrite=True)

# Launch the app
session = fo.launch_app(dataset)
