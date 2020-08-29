import numpy as np
import random
import math
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input, Dense, concatenate, Flatten, Add, Activation, BatchNormalization, Reshape, Permute
from tensorflow.keras import Model

from coord_tf import CoordinateChannel2D, CoordinateChannel1D

OUT_DIM = 20000


def cb(x, channel, kernel, padding='same'):
    x = Conv1D(channel, kernel_size=kernel, padding=padding)(x)
    x = BatchNormalization(gamma_initializer='zeros')(x)
    return x


def res(x, layers, kernel=(3,), act='relu', se=0, **kws):
    normalizer = BatchNormalization

    ConvLayer = k.layers.Conv1D
    MaxPoolingLayer = k.layers.MaxPooling1D
    AvePoolingLayer = k.layers.AveragePooling1D
    GlobalPoolingLayer = k.layers.GlobalAveragePooling1D
    GlobalMaxLayer = k.layers.GlobalMaxPooling1D
    assert K.ndim(x) == 3

    raw_x = x  # backup input

    x = ConvLayer(layers, kernel_size=kernel, padding='same', **kws)(x)
    x = normalizer(gamma_initializer='zeros')(x)

    if se == 1:
        x2 = GlobalPoolingLayer()(x)
        x2 = Dense(max(4, layers // 16), activation='relu')(x2)
        x2 = Dense(layers, activation='sigmoid')(x2)
        x2 = k.layers.Reshape((1, -1))(x2)

        x = k.layers.Multiply()([x, x2])

    if K.int_shape(x)[-1] != layers:
        raw_x = ConvLayer(layers, kernel_size=1, padding='same')(raw_x)
        raw_x = normalizer()(raw_x)

    x = Add()([raw_x, x])

    return Activation(act)(x)  # final activation


def build(act='elu'):
    inp = Input(shape=(32, 23))

    x = CoordinateChannel1D()(inp)  # add positional information

    features = k.layers.Concatenate(
        axis=-1)([cb(x, 64, i) for i in range(2, 10)])

    x = Conv1D(512, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, features])
    x = Activation(act)(x)

    for i in range(8):
        x = res(x, 512, 3, act=act, se=1)

    for i in range(3):
        x = res(x, 512, 1, se=0, act=act)

    x = k.layers.Conv1D(OUT_DIM, kernel_size=1, padding='valid')(x)
    x = Activation('sigmoid')(x)
    x = k.layers.GlobalAveragePooling1D(name='spectrum')(x)

    pmodel = k.models.Model(inputs=inp, outputs=x, name="predfull_model")
    return pmodel


pm = build()
pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss='cosine')
print(pm.summary())
