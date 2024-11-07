from prepare_data import create_dataset_from_imgs, read_file_list, shuffle_lists
from my_tensorboard import MyTensorBoard
from my_model import my_load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Input
from denoise_net import denoise_network, SKConv
from tensorflow.python.client import device_lib
import argparse
print(device_lib.list_local_devices())
import os
from my_loss import MSE, ssim_loss, ssim, total_loss, PSNR

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser(description='tensorflow implementation of denoising network')
parser.add_argument('--train_images', type=str, default=0,
                    help='Path of training input')
parser.add_argument('--train_labels', type=str, default=0,
                    help='Path of training label')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size')
parser.add_argument('--input_height', type=int, default=2,
                    help='the height of input image')
parser.add_argument('--input_width', type=int, default=2,
                    help='the width of input image')




if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size # batch_size = 4
    input_height = args.input_height # input_size = (320, 320, 1)
    input_width = args.input_width
    input_size = (input_height, input_width, 1)
    epochs = 5000
    logdir = r'logs'

    images = read_file_list(args.train_images)
    labels = read_file_list(args.train_labels)

    imgNum = len(images)
    idx1 = []
    idx2 = []
    for i in range(round(imgNum * 0.7)):
        idx1.append(i)
    for j in range(round(imgNum * 0.7), round(imgNum)):
        idx2.append(j)
    data_idx = [idx1, idx2]

    train_data = [images[i] for i in data_idx[0]]
    train_label = [labels[j] for j in data_idx[0]]
    valid_data = [images[i] for i in data_idx[1]]
    valid_label = [labels[j] for j in data_idx[1]]

    training_bscan, training_label = shuffle_lists(train_data, train_label)
    valid_bscan, valid_label = shuffle_lists(valid_data, valid_label)
    # create data generator
    my_training_batch_generator = create_dataset_from_imgs(training_bscan, training_label, batch_size, out_size=input_size)
    my_validation_batch_generator = create_dataset_from_imgs(valid_bscan, valid_label, batch_size, out_size=input_size)

    input_img = Input(shape=input_size)
    output = denoise_network(input_img, batch_size=batch_size, input_shape=(input_size[0], input_size[1]))
    model = Model(input_img, output)

    model.compile(optimizer=adam_v2.Adam(lr=0.001), loss=total_loss, metrics=[MSE, ssim_loss, ssim, PSNR])
    print(model.summary())

    tensorboard_visualization = MyTensorBoard(log_dir=logdir,
                                            write_graph=True, write_images=True)
    csv_logger = CSVLogger('{}/training.log'.format(logdir))
    checkpoint = ModelCheckpoint(logdir + '/model_{epoch:02d}_{val_loss:.2f}.ckpt', monitor='val_loss',
                                verbose=1, save_best_only=True)
    resuce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001,
                                cooldown=0, min_lr=0.00000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    callbacks = [tensorboard_visualization, checkpoint, resuce_lr, early_stopping, csv_logger]
    [model, init_epoch] = my_load_model(model, logdir=logdir, checkpoint_file='checkpoint.ckpt',
                                        custom_objects={'SKConv': SKConv})
    # training
    model.fit_generator(generator=my_training_batch_generator,
                        epochs=epochs,
                        initial_epoch=init_epoch,
                        steps_per_epoch=100,
                        validation_steps=1,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=my_validation_batch_generator)
