import argparse
import numpy as np
import pandas as pd
import math

import tensorflow.keras as k
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Add, Flatten, Activation, BatchNormalization, LayerNormalization
from tensorflow.keras import Model, Input

from coord_tf import CoordinateChannel2D, CoordinateChannel1D

from pyteomics import mgf, mass, mzid
cmass = mass

### Parameters

max_it = 1.0e4
min_it = 0.0

precision = 0.1
low = 0
dim = 20000
upper = math.floor(low + dim * precision)
mz_scale = 3000.0
max_mz = dim * precision + low

max_out = dim
it_scale = max_it

max_len = 30
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

def sparse(x, y, th = 0.005):
    y = scale(y)
    mz = []
    it = []
    for i in range(len(y)):
        if y[i] >= th:
            mz.append(x[i])
            it.append(y[i])
    return np.asarray(mz, dtype='float32'), np.asarray(it, dtype='float32')


mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538}

ave_mass = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 160.1598, "E": 129.1155,
            "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
            "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782, "T": 101.1051,
            "W": 186.2132, "Y": 163.1760, "V": 99.1326}

Alist = list('ACDEFGHIKLMNPQRSTVWY')
oh_dim = len(Alist) + 2

charMap = {'@': 0, '[': 21}
for i, a in enumerate(Alist): charMap[a] = i + 1

x_dim = oh_dim + 1
xshape = (max_in, x_dim)

# embed input item into a matrix
def embed(sp, mass_scale = max_mz, out=None, ignore=False, pep=None):
    if out is None: em = np.zeros((max_in, x_dim), dtype='float32')
    else: em = out

    if pep is None: pep = sp['pep']

    if len(pep) > max_len and ignore != False: return em # too long

    em[len(pep)][21] = 1 # ending pos, next line with +1 to skip this
    for i in range(len(pep) + 1, max_in - 1): em[i][0] = 1 # padding first, meta column should not be affected

    meta = em[-1]
    meta[0] = fastmass(pep, ion_type='M', charge=1) / mass_scale # pos 0, and overwrtie above padding
    meta[sp['charge']] = 1 # pos 1 - 4
    meta[5 + sp['type']] = 1 # pos 5 - 8
    meta[-1] = sp['nce'] / 100.0 if 'nce' in sp else 0.25

    for i in range(len(pep)):
        em[i][charMap[pep[i]]] = 1 # 1 - 20
        em[i][-1] = mono[pep[i]] / mass_scale

    return em

def f2(x): return "{0:.4f}".format(x)
def f4(x): return "{0:.4f}".format(x)

def tomgf(sp, y):
    head = ("BEGIN IONS\n"
        f"Title={sp['pep']}\n"
        f"CHARGE={sp['charge']}+\n"
        f"PEPMASS={sp['mass']}\n")

    imz = np.arange(0, dim, dtype='int32') * precision + low # more acurate

    mzs, its = sparse(imz, y)
    peaks = [f"{f2(mz)} {f4(it * 1000)}" for mz, it in zip(mzs, its)]

    return head + '\n'.join(peaks) + '\nEND IONS'

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input file path', default='example.tsv')
parser.add_argument('--output', type=str, help='output file path', default='example.mgf')
parser.add_argument('--model', type=str, help='model file path', default='pm.h5')

args = parser.parse_args()

K.clear_session()

pm = k.models.load_model(args.model, compile=0)
pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss='cosine')

inputs = []

# type: 0 unknown, 1 cid, 2 etd, 3 hcd
types = {'HCD': 3, 'ETD': 2}

for item in pd.read_csv(args.input, sep='\t').itertuples():
    if len(item.Peptide) <= max_len:
        inputs.append({'pep': item.Peptide, 'charge': item.Charge, 'type': types[item.Type],
                    'nce': item.NCE, 'mass': fastmass(item.Peptide, 'M', item.Charge)})

def input_generator(x, batch_size):
    while len(x) > batch_size:
        yield asnp32([embed(item) for item in x[:batch_size]])
        x = x[batch_size:]
    yield asnp32([embed(item) for item in x])

batch_size = 128
y = pm.predict(input_generator(inputs, batch_size), verbose=1, steps=int(math.ceil(len(inputs) / batch_size)))
y = np.square(y)

f = open(args.output, 'w+')
f.write('\n\n'.join([tomgf(sp, yi) for sp, yi in zip(inputs, y)]))
f.close()
