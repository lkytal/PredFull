import argparse
import numpy as np
import pandas as pd
import random
import math
import keras as k
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.layers import Conv1D, MaxPooling1D, Input, Dense, LSTM, concatenate, Flatten, Add, Subtract, Activation, BatchNormalization, UpSampling1D, Dropout,Reshape, Permute
from keras import Model
import tensorflow as tf

from coord import CoordinateChannel2D, CoordinateChannel1D
from position import Position_Embedding

# install cmass for better performance
# pip install pyteomics.cythonize
import pyteomics
from pyteomics import mgf, mass, mzid #, cmass
cmass = mass

### Parameters

max_it = 1.0e4
min_it = 0.0

precision = 0.1
low = 180.0
dim = 18200
upper = math.floor(low + dim * precision)
max_out = dim
max_mz = 2000

max_len = 22
max_in = max_len + 2
max_charge = 4

def pre(): return precision
def mz2pos(mz, pre=pre()): return int(round((mz - low) / pre))
def pos2mz(pos, pre=pre()): return pos * pre + low

def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype='float32')


def fastmass(pep, ion_type, charge):
    return cmass.fast_mass(pep, ion_type=ion_type, charge=charge) + 57.021 * pep.count('C') / charge

def normalize(it):
    it[it < 0] = 0
    return np.sqrt(np.sqrt(it / max_it))

def scale(v, _max_it=1.0, inplace = False):
    c0 = np.max(v)
    if c0 == _max_it or c0 == 0: return v #no need to scale

    c = _max_it / c0
#     return v * c
    if inplace: return np.multiply(v, c, dtype='float32', out=v)
    else: return np.multiply(v, c, dtype='float32')

def sparse(x, y, th = 0.05):
    y = scale(y)
    mz = []
    it = []
    for i in range(len(y)):
        if y[i] >= th:
            mz.append(x[i])
            it.append(y[i])
    return np.asarray(mz, dtype='float32'), np.asarray(it, dtype='float32')

mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768, "C": 160.03019,
        "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04048, "m": 147.0354,
        "H": 137.05891, "F": 147.06441, "R": 156.10111, "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538}

ave = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 160.1598, "E": 129.1155, "Q": 128.1307,
       "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766,
       "P": 97.1167, "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326}

charMap = { "A": 1, "R": 2, "N": 3, "D": 3, "C": 5, "E": 5, "Q": 7,
           "G": 8, "H": 9, "I": 10, "L": 11, "K": 12, "M": 13, "F": 14,
           "P": 15, "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20 }

x_dim = 20 + 2 + 3

def embed(sp, mass_scale = max_mz):
    em = np.zeros((max_in, x_dim), dtype='float32')
    # pep = pep.replace('I', 'L')
    pep = sp['pep']
    meta = em[-1]

    em[len(pep)][21] = 1 # ending pos, next line with +1 to skip this
    for i in range(len(pep) + 1, max_in): em[i][0] = 1 # padding first, as meta column should not be affected

    meta[0] = fastmass(pep, ion_type='M', charge=1) / mass_scale # pos 0, and overwrtie above padding
    meta[sp['charge']] = 1 # pos 1 - 4
    meta[5 + sp['type']] = 1 # pos 5 - 8

    meta[9] = sp['nce'] / 100.0 if 'nce' in sp else 0.25

    mass1c = fastmass(pep, ion_type='M', charge=1) # total mass of 1+ M ion
    for i in range(len(pep)):
        em[i][charMap[pep[i]]] = 1 # 1 - 20
        em[i][-1] = mono[pep[i]] / mass_scale

        b_mass = fastmass(pep[:i], ion_type='b', charge=1) # just embed +1 ions
        em[i][-2] = b_mass / mass_scale
        em[i][-3] = (mass1c - b_mass + 1.00794) / mass_scale

    return em

def cb(x, channel, kernel, pad='same', zero=0):
    param = {} if zero == 0 else {"gamma_initializer" :'zeros'}
    x = Conv1D(channel, kernel_size=kernel, padding=pad)(x)
    x = BatchNormalization(**param)(x)
    return x

def res(x1, layers, kernel=(3,), act='relu', identity=True, se=0, **kws):
    normalizer = BatchNormalization
    param = {"gamma_initializer" :'zeros'}

    ConvLayer = k.layers.Conv1D
    MaxPoolingLayer = k.layers.MaxPooling1D
    AvePoolingLayer = k.layers.AveragePooling1D
    GlobalPoolingLayer = k.layers.GlobalAveragePooling1D
    GlobalMaxLayer = k.layers.GlobalMaxPooling1D
    assert K.ndim(x1) == 3

    c1 = x1

    c1 = ConvLayer(layers, kernel_size=kernel, use_bias=0, padding='same', **kws)(c1)
    c1 = normalizer(**param)(c1)

    if se == 1:
        s1 = GlobalPoolingLayer()(c1)
        s1 = Dense(max(4, layers // 16), activation='relu')(s1)
        s1 = Dense(layers, activation='sigmoid')(s1)
        s1 = k.layers.Reshape((1, -1))(s1)

        c1 = k.layers.Multiply()([c1, s1])

    o1 = x1
    if identity == False or K.int_shape(x1)[-1] != layers:
        o1 = ConvLayer(layers, kernel_size=1, use_bias=0, padding='same')(o1)
        o1 = normalizer()(o1) # no gamma, main path

    v1 = Add()([c1, o1])

    v1 = Activation(act)(v1) #final activation

    return v1

def build(act=lambda : 'elu'):
    inp = Input(shape=(max_in, x_dim))
    v1 = inp
    outlayers = []

    v1 = Position_Embedding(8, mode='concat')(v1)
    v1 = CoordinateChannel1D()(v1)

    o1 = v1 # preserve
    feathers = [cb(v1, 64, i, zero=1) for i in range(2, 10)]
    v1 = k.layers.Concatenate(axis=-1)(feathers)

    o1 = Conv1D(512, kernel_size=1, padding='same')(o1)
    o1 = BatchNormalization()(o1)

    v1 = Add()([v1, o1])
    v1 = Activation(act())(v1)

    for i in range(8): v1 = res(v1, 512, 3, act=act(), se=1)

    for i in range(3): v1 = res(v1, 512, 1, se=0, act=act())

    v1 = k.layers.Conv1D(max_out, kernel_size=1, padding='valid')(v1)
    v1 = Activation('sigmoid')(v1) #last
    v1 = k.layers.GlobalAveragePooling1D(name='spectrum')(v1)

    pmodel = k.models.Model(inputs=inp, outputs=v1, name="pred_model")
    return  pmodel

K.clear_session()

pm = build()
pm.compile(optimizer=k.optimizers.adam(lr=0.0003), loss='cosine')
pm.load_weights('pm.hdf5')

def f2(x): return "{0:.4f}".format(x)
def f4(x): return "{0:.4f}".format(x)

def tomgf(sp, y):
    head = ("BEGIN IONS\n"
        f"Title={sp['pep']}\n"
        f"CHARGE={sp['charge']}+\n"
        f"PEPMASS={sp['mass']}\n")

    imz = np.arange(0, dim, dtype='int32') * precision + low # more acurate

    y = y ** 4 # re
    mzs, its = sparse(imz, y, th=0.001)
    peaks = [f"{f2(mz)} {f4(it * 1000)}" for mz, it in zip(mzs, its)]

    return head + '\n'.join(peaks) + '\nEND IONS'

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input file path', default='example.tsv')
parser.add_argument('--output', type=str, help='output file path', default='example.mgf')
parser.add_argument('--weight', type=str, help='weight file path', default='pm.hdf5')

args = parser.parse_args()

sps = []

# type: 0 unknown, 1 cid, 2 etd, 3 hcd
types = {'HCD': 3, 'ETD': 2}

for item in pd.read_csv(args.input, sep='\t').itertuples():
    if len(item.Peptide) <= max_len:
        sps.append({'pep': item.Peptide, 'charge': item.Charge, 'type': types[item.Type],
                    'nce': item.NCE, 'mass': fastmass(item.Peptide, 'M', item.Charge)})

x = [embed(sp) for sp in sps]
y = pm.predict(asnp32(x))

f = open(args.output, 'w+')
f.write('\n\n'.join([tomgf(sp, yi) for sp, yi in zip(sps, y)]))
f.close()
