import os
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime
#
IMG_DIR = '../data/images_ukraine/'
IMG_PROCESSED_DIR = '../data/images_ukraine_extracted/'

# make sure the directory exists
os.makedirs(IMG_PROCESSED_DIR, exist_ok=True)

def unzip_and_cleanup(dir_name):
    dir_path = os.path.join(IMG_DIR, dir_name)
    if os.path.isfile(dir_path):
        return

    for file in os.listdir(dir_path):
        if file.endswith('.zip'):
            # time = datetime.now().time().strftime("%H:%M:%S")
            # print(f"{time}-{file}: Processing...")
            zip_path = os.path.join(dir_path, file)
            img_name = file.split('.')[0]
            extract_to = os.path.join(IMG_PROCESSED_DIR, img_name)
            unzip_file(zip_path, extract_to, file)

    
    shutil.rmtree(dir_path)
    # time = datetime.now().time().strftime("%H:%M:%S")
    # print(f"{time}-{dir_name}: removed...")

def unzip_file(zip_path, extract_to, file):
    # Ensure the extraction directory exists
    os.makedirs(extract_to, exist_ok=True)
    # time = datetime.now().time().strftime("%H:%M:%S")
    # print(f"{time}-{file}: created dir...")
    
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all contents to the specified directory
        zip_ref.extractall(extract_to)
    # time = datetime.now().time().strftime("%H:%M:%S")
    # print(f"{time}-{file}: extracted...")

if __name__ == "__main__":

    # List all directories in IMG_DIR
    dir_list = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]

    for dir_name in tqdm(dir_list, desc="Processing directories"):
        try:
            unzip_and_cleanup(dir_name)
        except Exception as e:
            print(e)


    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Wrap the executor map with tqdm for progress bar
        list(tqdm(executor.map(unzip_and_cleanup, dir_list), total=len(dir_list),
                  desc="Processing directories",
                  mininterval=1))

