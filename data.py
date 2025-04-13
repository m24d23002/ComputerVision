import os
import urllib.request
import zipfile


def download_and_extract(url, extract_path):
    # Extract the filename from the URL
    filename = url.split("/")[-1]
    
    # Create the 'COCO' folder if it doesn't exist
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    # Check if the file already exists
    file_path = os.path.join(extract_path, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)
    else:
        print(f"{filename} already downloaded.")
    
    # Extract the ZIP file
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Done extracting to {extract_path}")


# Define the folder where you want to store COCO data
data_dir = './COCO'


# Download and extract COCO val2017 images
download_and_extract(
    "http://images.cocodataset.org/zips/val2017.zip", 
    data_dir
)


# Download and extract COCO annotations
download_and_extract(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 
    data_dir
)
