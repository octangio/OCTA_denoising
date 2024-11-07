from tensorflow.python.keras import backend as K
import tensorflow as tf
import cv2
import numpy as np


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def MAE(y_true, y_pred):
    MAE_loss = K.mean(K.abs(y_pred - y_true), axis=-1)
    return MAE_loss


def MSE(y_true, y_pred):
    MSE_loss = tf.reduce_mean(K.mean(K.square(y_pred - y_true), axis=-1))
    return MSE_loss


def ssim(y_true, y_pred):
    SSIM = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return SSIM


def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 1.0)
    dessim = 1 - ssim
    return dessim

def total_loss(y_true, y_pred):
    MSE_loss = MSE(y_true, y_pred)
    SSIM_loss = ssim_loss(y_true, y_pred)
    add_loss = MSE_loss + SSIM_loss
    return add_loss

