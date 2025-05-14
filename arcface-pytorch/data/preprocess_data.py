import cv2 
import multiprocessing
import os
import numpy as np
from tqdm.auto import tqdm

def sampling_images_path(input_dir: str, num_samples= 30):
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if len(image_files) < num_samples or num_samples < 1:
        sampled_files = image_files
    else:
        sampled_files = np.random.choice(image_files, num_samples, replace=False)

    sampled_files = [os.path.join(input_dir, f) for f in sampled_files]
    return sampled_files

def get_data_list(input_dir: str, input_file, output_file: str, num_samples= -1):
    with open(output_file, 'w') as f:
        with open(input_file, 'r') as fd:
            lines = fd.readlines()

        result_lines = []
        for folder_and_gender in lines:
            folder, gender = folder_and_gender.strip().split()
            folder = os.path.basename(folder)
            gender = "M" if "male" == gender.strip().lower() else "F" if "female" == gender.strip().lower() else gender
            folder_path = os.path.join(input_dir, folder)
            if os.path.isdir(folder_path):
                image_files = sampling_images_path(folder_path, num_samples)
                images_file_str = "\t".join(image_files)
                images_file_str = folder + "\t" + gender + "\t" + images_file_str + "\n"
                result_lines.append(images_file_str)
        f.writelines(result_lines)

    print(f"Data list saved to {output_file}")


def split_train_test(input_path: str, train_ratio = 0.8, class_balance_on_test = True):
    with open(input_path, 'r') as fd:
        lines = fd.readlines()
    total_cls = len(lines)
    train_cls = int(total_cls * train_ratio)
    test_cls = total_cls - train_cls

    train_samples_list = []
    test_samples_list = []

    if class_balance_on_test:
        cls_labels = []
        for line in lines:
            class_label = line.split()[1]
            cls_labels.append(class_label)
        cls_labels = np.array(cls_labels)
        idx = np.arange(len(lines))
        unique_labels, count = np.unique(cls_labels, return_counts=True)
        min_count = np.min(count).astype(np.int32).item()
        argmin_count = np.argmin(count)

        if min_count/2 > test_cls/len(unique_labels):
            sample_to_take = int(test_cls/len(unique_labels))
        else:
            sample_to_take = int(min_count/2)

        for cls, count in zip(unique_labels, count):
            samples = lines[idx[cls_labels == cls]]
            samples = np.random.permutation(samples)
            test_samples = samples[:sample_to_take]
            train_samples = samples[sample_to_take:]
            train_samples_list.extend(train_samples.tolist())
            test_samples_list.extend(test_samples.tolist())
    else:
        lines = np.random.permutation(lines)
        train_samples_list = lines[:train_cls]
        test_samples_list = lines[train_cls:]

    with open(input_path.replace(".txt", "_train.txt"), 'w') as f:
        f.writelines(train_samples_list)
    with open(input_path.replace(".txt", "_test.txt"), 'w') as f:
        f.writelines(test_samples_list)


        