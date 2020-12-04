import os
import numpy as np


def list_of_subjects(data_dir, extension):
    """
    Returns the list of subjects in a directory.

    input:
    data_dir: directory name (eg: "/analysis/ritter/data/data_dir")
    extension: file extension to look for(eg: ".mat")

    """

    file_list = []
    for files in os.listdir(data_dir):
        if files.endswith(extension):
            file_list.append(files)
    return file_list


def write_list_to_file(file_name, list_name):
    """
    Writes a list to a file.

    input:
    file_name: The file name you want to give
    list_name: List from where data needs to be taken.

    """
    with open(file_name, 'w') as f:
        for item in list_name:
            f.write("%s\n" % item)


def check_common_subjects(list1, list2):
    """
    To make sure you dont have common subjects in the split made.
    list1 : could be one of these (list of train subjects, test subjects, validation subjects)
    list2 : could be one of these (list of train subjects, test subjects, validation subjects)
    """

    train_list = set(list1)
    test_list = set(list2)
    if (train_list & test_list):
        print(train_list & test_list)
    else:
        print("There are no common subjects")

def split_data(data, age_count, gender_count, gender, age):
    class_counts_gender = {}
    class_counts_age = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    train_class = {}
    test_class = {}

    for index, label in enumerate(age):
        c = label
        g = gender[index]
        class_counts_age[c] = class_counts_age.get(c, 0) + 1
        class_counts_gender[g] = class_counts_gender.get(g, 0) + 1
        if class_counts_age[c] > 1 and class_counts_age[c] <= age_count and class_counts_gender[g] <= gender_count:
            test_data.append(data[index])
            test_label.append(label)
            test_class[c] = test_class.get(c, 0) + 1
            test_class[g] = test_class.get(g, 0) + 1
        else:
            train_data.append(data[index])
            train_label.append(label)
            train_class[c] = train_class.get(c, 0) + 1
            train_class[g] = train_class.get(g, 0) + 1

    return train_data, test_data


def save_as_numpy_array(dest_dir, subject, data):
    """
    Save the data to a numpy array
    """
    file_name = subject + ".npz"
    np.savez(os.path.join(dest_dir, file_name), data)
