# from image_search_engine.data.jumia_image_dataset_v1 import JumiaImageDataset
# from image_search_engine import data


# print("Success")


import os


def check_init_files(directory):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            init_file_path = os.path.join(root, dir, "__init__.py")
            if not os.path.isfile(init_file_path):
                return False
    return True


def find_submodules_without_init(directory):
    missing_init = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            init_file_path = os.path.join(root, dir, "__init__.py")
            if not os.path.isfile(init_file_path):
                missing_init.append(os.path.join(root, dir))
    return missing_init


import os
import shutil


def remove_pycache(directory):
    for root, dirs, files in os.walk(directory):
        if "__pycache__" in dirs:
            pycache_dir = os.path.join(root, "__pycache__")
            print(f"Removing {pycache_dir}")
            shutil.rmtree(pycache_dir)


print(find_submodules_without_init("image_search_engine"))
