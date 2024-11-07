from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, add, concatenate
import numpy as np
import os
from denoise_net import denoise_network
import cv2
import scipy.io as sio
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Tensorflow implementation of denoising network')
parser.add_argument('--test_data_path', type=str, default=0,
                    help='Path of test input image')
parser.add_argument('--save_path', type=str, default=0,
                    help='The folder path to save output')
parser.add_argument('--logdir', type=str, default=0,
                    help='Path of model weight')


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.test_data_path
    save_path = args.save_path
    log_dir = args.logdir

    batch_size = 1

    input_img = Input(shape=(None, None, 1))
    output = denoise_network(input_img, batch_size=batch_size, input_shape=(512, 512))
    model = Model(input_img, output)
    model.load_weights(log_dir)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_list = os.listdir(data_path)
    for i in range(0, len(file_list)):
        print(i + 1)
        image = data_path + file_list[i]
        name = str(file_list[i])

        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        if shape[0] % 16 != 0 or shape[1] % 16 != 0:
            new_w = (shape[1] // 16) * 16
            new_h = (shape[0] // 16) * 16
            resize_img = cv2.resize(img, (new_w, new_h))
        else:
            resize_img = img

        new_shape = resize_img.shape
        Y = np.expand_dims(resize_img, 0)
        Y = np.expand_dims(Y, 3) / 255.

        pre = model.predict(Y, batch_size=1) * 255.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)
        pre = np.squeeze(pre)
        out_name = name.split('.png')[0] + '_out.png'
        output_img = cv2.resize(pre, (shape[1], shape[0]))
        OUTPUT_NAME = save_path + out_name
        cv2.imwrite(OUTPUT_NAME, output_img)
