import tensorflow as tf
from tensorflow.python.keras import layers, regularizers
from tensorflow.python.keras.layers import Conv2D, add, Activation, UpSampling2D, BatchNormalization
from tensorflow.python.keras import layers
import numpy as np


class SKConv(layers.Layer):
    def __init__(self, M=2, r=16, L=32, G=32, batch_size=2, input_shape=(2, 2), channel=1, **kwargs):
        super(SKConv, self).__init__(**kwargs)
        self.M = M
        self.r = r
        self.L = L
        self.G = G

        b, h, w = batch_size, input_shape[0], input_shape[1]
        filters = channel
        d = max(filters // self.r, self.L)

        self.x_layersList = []
        for m in range(1, self.M + 1):
            x_layers = tf.keras.Sequential()
            if self.G == 1:
                x_layers.add(layers.Conv2D(filters, 3, dilation_rate=m, padding='same',
                                           use_bias=False, name=self.name + '_conv%d' % m))
            else:
                c = filters // self.G
                x_layers.add(layers.DepthwiseConv2D(3, dilation_rate=m, depth_multiplier=c, padding='same',
                                                    use_bias=False, name=self.name + '_conv%d' % m))

                x_layers.add(layers.Reshape([h, w, self.G, c, c], name=self.name + '_conv%d_reshape1' % m))
                x_layers.add(layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1),
                                           output_shape=[b, h, w, self.G, c],
                                           name=self.name + '_conv%d_sum' % m))
                x_layers.add(layers.Reshape([h, w, filters],
                                            name=self.name + '_conv%d_reshape2' % m))

            # x_layers.add(BatchNormalization(name=self.name + '_conv%d_bn' % m))
            x_layers.add(layers.Activation('relu', name=self.name + '_conv%d_relu' % m))
            self.x_layersList.append(x_layers)

        self.add_x_layers = layers.Add(name=self.name + '_add')
        self.reduce_mean = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True),
                                         output_shape=[b, 1, 1, filters],
                                         name=self.name + '_gap')

        self.cov_z = layers.Conv2D(d, 1, name=self.name + '_fc_z')
        # self.bn = BatchNormalization(name=self.name + '_fc_z_bn')
        self.z_active = layers.Activation('relu', name=self.name + '_fc_z_relu')

        self.cov_x = layers.Conv2D(filters * self.M, 1, name=self.name + '_fc_x')
        self.reshape_x = layers.Reshape([1, 1, filters, self.M], name=self.name + '_reshape')
        self.scale = layers.Softmax(name=self.name + '_softmax')

        self.lamda_x = layers.Lambda(lambda x: tf.stack(x, axis=-1),
                                     output_shape=[b, h, w, filters, self.M],
                                     name=self.name + '_stack')  # (xs)  # b, h, w, c, M
        self.lamda_x1 = layers.Lambda(
            lambda x: tf.reduce_sum(tf.multiply(x[0], x[1], name='product'), axis=-1, name='rdcsum'),
            output_shape=[b, h, w, filters], name=self.name + "_multiply")  # ((scale, x))

    def build(self, input_shape):
        super(SKConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        xs = []
        for x_layers in self.x_layersList:
            xs.append(x_layers(inputs))
        x = self.add_x_layers(xs)
        x = self.reduce_mean(x)
        x = self.cov_z(x)
        x = self.z_active(x)
        x = self.cov_x(x)
        x = self.reshape_x(x)
        xsc = self.scale(x)
        x = self.lamda_x(xs)
        x = self.lamda_x1((xsc, x))
        return x

    def get_config(self):
        config = super(SKConv, self).get_config()
        config.update({})
        return {'M': self.M, 'r': self.r, 'L': self.L, 'G': self.G}


def conv2d_bn(input_tensor, filters, kernel_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), use_bias=False,
              kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(5e-4), activation=None, name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv2d'
        # bn_name = name + '_bn'
        conv2d_name = name + '_conv2d'
        act_name = name + '_atv'
    else:
        # bn_name = None
        conv_name = 'conv2d'
        act_name = None
        conv2d_name = None
    with tf.name_scope(name=conv_name):
        xi = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dilation_rate=dilation_rate,
            name=conv2d_name)(input_tensor)
        if activation is not None:
            xi = layers.Activation(activation, name=act_name)(xi)
    return xi


def dense_block(x, layer_num=3):
    fea = x
    for i in range(layer_num):
        conv = conv2d_bn(fea, 64, (3, 3), strides=(1, 1), activation='relu')
        fea = layers.Concatenate()([fea, conv])
    return fea


def denoise_network(x, batch_size=2, input_shape=(2, 2)):
    with tf.device('gpu:0'):
        conv1 = conv2d_bn(x, 32, (3, 3), strides=(1, 1), activation='relu')
        dense_fea1 = dense_block(conv1, layer_num=5)

        fea_d2 = conv2d_bn(dense_fea1, 64, (3, 3), strides=(2, 2), activation='relu')
        dense_fea2 = dense_block(fea_d2, layer_num=5)

        fea_d4 = conv2d_bn(dense_fea2, 128, (3, 3), strides=(2, 2), activation='relu')
        dense_fea4 = dense_block(fea_d4, layer_num=5)
        # fea_d8 = conv2d_bn(dense_fea4, 128, (3, 3), strides=(2, 2), activation='relu')
        # dense_fea8 = dense_block(fea_d8, layer_num=5)


    # with tf.device('gpu:1'):
        dense_fea_sk = SKConv(3, G=1, batch_size=batch_size, input_shape=list(np.array(input_shape)),
                              channel=192, name='skconv1')(dense_fea1)
        dense_fea2_sk = SKConv(3, G=1, batch_size=batch_size, input_shape=list(np.array(input_shape) // 2),
                               channel=384,
                               name='skconv2')(
            dense_fea2)
    with tf.device('gpu:1'):
        dense_fea4_sk = SKConv(3, G=1, batch_size=batch_size, input_shape=list(np.array(input_shape) // 4),
                               channel=768,
                               name='skconv3')(
            dense_fea4)
        # dense_fea8_sk = SKConv(3, G=1, batch_size=batch_size, input_shape=list(np.array(input_shape) // 8),
        #                        channel=768,
        #                        name='skconv4')(
        #     dense_fea8)
        # dense_fea8_u8 = UpSampling2D((8, 8))(dense_fea8_sk)
        dense_fea4_u4 = UpSampling2D((4, 4))(dense_fea4_sk)
        dense_fea2_u2 = UpSampling2D((2, 2))(dense_fea2_sk)

        feature = layers.Concatenate()([dense_fea_sk, dense_fea2_u2, dense_fea4_u4])
        out_layer1 = conv2d_bn(feature, 512, (3, 3), strides=1, activation='relu', padding='same')
        out_layer = conv2d_bn(out_layer1, 1, (3, 3), strides=1, activation=None, padding='same')
        return out_layer


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    inputs = Input([200, 200, 3])
    # x = SKConv(3, G=1)(inputs)
    x = denoise_network(inputs, batch_size=2, input_shape=(200, 200))
    m = Model(inputs, x)
    m.summary()

    import numpy as np

    X = np.random.random([2, 200, 200, 3]).astype(np.float32)
    y = m.predict(X)
    print(y.shape)
