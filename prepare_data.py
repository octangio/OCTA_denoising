import tensorflow as tf
import numpy as np
import cv2
import json
import os
import sys
import scipy.io as sio
import PIL.Image as Image
import matplotlib.pyplot as plt

"""
Better performance with the tf.data
files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2, num_parallel_calls=tf.data.experimental.AUTOTUNE) # This is used to remotly stored data
       .prefetch(tf.data.experimental.AUTOTUNE)

       .map(time_consuming_mapping_fuc,num_parallel_calls=tf.data.experimental.AUTOTUNE)
       .cache()
       .map(memory_consuming_mapping_fuc)
       .shuffle(100)
       .batch(2)
       .repeat()
"""


def read_image(image_file_name, color_mode=None):
    """
     Read image from file.
    :param image_file_name: full file path
    :param image_size_row_col: [row, col]
    :param color_mode: 'gray', 'rgb' or 'idx'
    :return: numpy image
    """
    img = Image.open(image_file_name.numpy().rstrip())
    if color_mode is not None:
        color_mode = color_mode.numpy()
        if color_mode.lower() == 'gray':
            img = img.convert('L')
        else:

            if color_mode.lower() == 'rgb':
                img = img.convert('RGB')
            else:
                if color_mode.lower() == 'idx':
                    img = img.convert('P')
    return np.array(img)


def tf_read_image(image_file_name, color_model='idx'):
    img = tf.py_function(read_image, [image_file_name, color_model], [tf.float32])
    return tf.transpose(img, [1, 2, 0])


def read_file_list(list_txt_file):
    fp = open(list_txt_file, 'r')
    files = fp.readlines()
    files = [item.rstrip() for item in files]
    return files


def shuffle_lists(*file_lists):
    """
    Example:
        list1=['a', 'b', 'c']
        list2=[1, 2, 3]
        list1_s,list2_s=shuffle_data_files(list1,list2)
        list1_s = ['a', 'c', 'b']
        list2_s = [1, 3, 2]
    :param file_lists: any numbers of list
    :return: shuffled lists
    """
    if len(file_lists) == 1:
        list_files = list(*file_lists)
        np.random.shuffle(list_files)
        return list_files
    else:
        list_files = list(zip(*file_lists))
        np.random.shuffle(list_files)
        return zip(*list_files)


def create_dataset_from_imgs(filenames, labelname, batch_size=2, out_size=(320, 320, 1)):
    # filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
    # labels = ["/var/data/label1.png", "/var/data/label2.png", ...]

    # read image file
    def _tf_load_image(filename, labelname):
        out_image = tf_read_image(filename, 'gray') / 255.0
        out_label = tf_read_image(labelname, 'gray') / 225.0
        return out_image, out_label

    #  random crop image
    def _tf_random_crop(image_decoded, label_decoded):
        stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, *out_size])
        return cropped_image[0], cropped_image[1]

    # random flip left-right
    def _tf_random_flip_left_right(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.flip_left_right(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded

    # random flip up-down
    def _tf_random_flip_up_down(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.flip_up_down(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded

    def _tf_random_transpose(image_decoded, label_decoded):
        if tf.random.uniform(()) > 0.5:
            stacked_image = tf.stack([image_decoded, label_decoded], axis=0)
            fliped_image = tf.image.transpose(stacked_image)
            return fliped_image[0], fliped_image[1]
        else:
            return image_decoded, label_decoded


    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(filenames), tf.convert_to_tensor(labelname)))
    dataset = dataset.map(_tf_load_image)
    dataset = dataset.map(_tf_random_crop)
    dataset = dataset.map(_tf_random_transpose)
    dataset = dataset.map(_tf_random_flip_up_down)
    dataset = dataset.map(_tf_random_flip_left_right)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=30)
    dataset = dataset.batch(batch_size)
    return dataset