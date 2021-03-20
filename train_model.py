#!/usr/bin/env python3

import numpy as np
import math
from pyteomics import mgf, mass
import argparse

import tensorflow.keras as k
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Add, Flatten, Activation, BatchNormalization
from tensorflow.keras import Model, Input

from coord_tf import CoordinateChannel2D, CoordinateChannel1D

# hyper parameter and constants

SPECTRA_DIMENSION = 20000
BIN_SIZE = 0.1
MAX_PEPTIDE_LENGTH = 30
MAX_MZ = 2000


mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538}

ave_mass = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 160.1598, "E": 129.1155,
            "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
            "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782, "T": 101.1051,
            "W": 186.2132, "Y": 163.1760, "V": 99.1326}

Alist = list('ACDEFGHIKLMNPQRSTVWY')
encoding_dimension = len(Alist) + 2

charMap = {'@': 0, '[': 21}
for i, a in enumerate(Alist):
    charMap[a] = i + 1

# help functions

def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype='float32')

# compute percursor mass
def fastmass(pep, ion_type, charge, mod=None, cam=True):
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)

    if cam:
        base += 57.021 * pep.count('C') / charge

    if not mod is None:
        base += 15.995 * np.sum(mod == 1) / charge

        base += -np.sum(mod[mod < 0])
    return base

# embed input item into a matrix
def embed(sp, mass_scale=MAX_MZ):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float32')

    pep = sp['pep']

    if len(pep) > MAX_PEPTIDE_LENGTH:
        raise "input too long"

    # meta information
    meta = encoding[-1]
    meta[0] = fastmass(pep, ion_type='M', charge=1) / mass_scale
    meta[sp['charge']] = 1  # pos 1 - 4
    meta[8] = 1  # HCD type
    meta[-1] = sp['nce'] / 10000.0 if 'nce' in sp else 0.0025

    # one-hot encoding
    for i in range(len(pep)):
        encoding[i][charMap[pep[i]]] = 1  # 1 - 20
        encoding[i][-1] = mono[pep[i]] / mass_scale

    encoding[len(pep)][21] = 1  # ending charactor
    for i in range(len(pep) + 1, MAX_PEPTIDE_LENGTH - 1):
        encoding[i][0] = 1  # padding charactors

    return encoding

# read inputs
def parse_spectra(sps):
    # ratio constants for NCE
    cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

    db = []

    for sp in sps:
        param = sp['params']

        c = int(str(param['charge'][0])[0])

        if 'seq' in param:
            pep = param['seq']
        else:
            pep = param['title']

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])

        if 'hcd' in param:
            try:
                hcd = param['hcd']
                if hcd[-1] == '%':
                    hcd = float(hcd)
                elif hcd[-2:] == 'eV':
                    hcd = float(hcd[:-2])
                    hcd = hcd * 500 * cr[c] / mass
                else:
                    raise Exception("Invalid type!")
            except:
                hcd = 0
        else:
            hcd = 0

        mz = sp['m/z array']
        it = sp['intensity array']

        db.append({'pep': pep, 'charge': c,
                   'mass': mass, 'mz': mz, 'it': it, 'nce': hcd})

    return db


def readmgf(fn):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)

    codes = parse_spectra(data)
    file.close()
    return codes


def spectrum2vector(mz_list, itensity_list, mass, bin_size, charge):
    itensity_list = itensity_list / np.max(itensity_list)

    vector = np.zeros(SPECTRA_DIMENSION, dtype='float32')

    mz_list = np.asarray(mz_list)

    indexes = mz_list / bin_size
    indexes = np.around(indexes).astype('int32')

    for i, index in enumerate(indexes):
        vector[index] += itensity_list[i]

    # normalize
    vector = np.sqrt(vector)

    # remove precursors, including isotropic peaks
    for delta in (0, 1, 2):
        precursor_mz = mass + delta / charge
        if precursor_mz > 0 and precursor_mz < 2000:
            vector[round(precursor_mz / bin_size)] = 0

    return vector

# building the model
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


def build(act='relu'):
    inp = Input(shape=(MAX_PEPTIDE_LENGTH + 2, encoding_dimension))

    x = CoordinateChannel1D()(inp)  # add positional information

    features = k.layers.Concatenate(axis=-1)([cb(x, 64, i) for i in range(2, 10)])

    x = Conv1D(512, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, features])
    x = Activation(act)(x)

    for i in range(8):
        x = res(x, 512, 3, act=act, se=1)

    for i in range(3):
        x = res(x, 512, 1, se=0, act=act)

    x = k.layers.Conv1D(SPECTRA_DIMENSION, kernel_size=1, padding='valid')(x)
    x = Activation('sigmoid')(x)
    x = k.layers.GlobalAveragePooling1D(name='spectrum')(x)

    pmodel = k.models.Model(inputs=inp, outputs=x, name="predfull_model")
    return pmodel

parser = argparse.ArgumentParser()
parser.add_argument('--mgf', type=str,
                    help='output file path', default='hcd_testingset.mgf')
parser.add_argument('--out', type=str,
                    help='filename to save the trained model', default='trained.h5')

args = parser.parse_args()

K.clear_session()

pm = build()
pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss='cosine_similarity')
print(pm.summary())


print('Reading mgf...', args.mgf)
spectra = readmgf(args.mgf)

y = [spectrum2vector(sp['mz'], sp['it'], sp['mass'], BIN_SIZE, sp['charge']) for sp in spectra]

x = [embed(sp) for sp in spectra]

pm.fit(x=asnp32(x), y=asnp32(y), epochs=50, verbose=1)
pm.save(args.out)
