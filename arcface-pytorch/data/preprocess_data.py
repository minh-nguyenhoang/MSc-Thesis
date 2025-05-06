import cv2 
import multiprocessing
import os
import numpy as np
from tqdm.auto import tqdm

def sampling_images_path(input_dir: str, num_samples= 30):
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if len(image_files) < num_samples:
        sampled_files = image_files
    else:
        sampled_files = np.random.choice(image_files, num_samples, replace=False)

    sampled_files = [os.path.join(input_dir, f) for f in sampled_files]
    return sampled_files

def get_data_list(input_dir: str, input_file, output_file: str, num_samples= 30):
    with open(output_file, 'w') as f:
        with open(input_file, 'r') as fd:
            lines = fd.readlines()

        result_lines = []
        for folder_and_gender in lines:
            folder, gender = folder_and_gender.strip().split()
            folder = os.path.basename(folder)
            gender = "M" if "male" == gender.lower() else "F" if "female" == gender.lower() else gender
            folder_path = os.path.join(input_dir, folder)
            if os.path.isdir(folder_path):
                image_files = sampling_images_path(folder_path, num_samples)
                images_file_str = "\t".join(image_files)
                images_file_str = folder + "\t" + images_file_str + gender + "\n"
                result_lines.append(images_file_str)
        f.writelines(result_lines)

    print(f"Data list saved to {output_file}")


def split_train_test(input_path: str, class_balance_on_test = True):
    with open(input_path, 'r') as fd:
        lines = fd.readlines()

    if class_balance_on_test:
        cls_labels = []
        for line in lines:
            class_label = line.split()[-1]
            cls_labels.append(class_label)
        unique_labels, count = np.unique(cls_labels, return_counts=True)