import argparse
import numpy as np
import pandas as pd
import math
from pyteomics import mgf, mass

import tensorflow.keras as k
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Add, Flatten, Activation, BatchNormalization, LayerNormalization
from tensorflow.keras import Model, Input

from coord_tf import CoordinateChannel2D, CoordinateChannel1D

# Parameters

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

# help functions


def pre(): return precision
def mz2pos(mz, pre=pre()): return int(round((mz - low) / pre))
def pos2mz(pos, pre=pre()): return pos * pre + low


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


def sparse(x, y, th=0.005):
    x = np.asarray(x, dtype='float32')
    y = np.asarray(y, dtype='float32')

    y /= np.max(y)

    return x[y > th], y[y > th]

# help function to parse modifications


def getmod(pep):
    mod = np.zeros(len(pep))

    if pep.isalpha():
        return pep, mod, 0

    seq = []
    nmod = 0

    i = -1
    while len(pep) > 0:
        if pep[0] == '(':
            if pep[:3] == '(O)':
                mod[i] = 1
            else:
                mod[i] = -2

            pep = pep[3:]
        elif pep[0] == '+' or pep[0] == '-':
            sign = 1 if pep[0] == '+' else -1

            for j in range(1, len(pep)):
                if pep[j] not in '.1234567890':
                    if i == -1:  # N-term mod
                        nmod += sign * float(pep[1:j])
                    else:
                        mod[i] += sign * float(pep[1:j])
                    pep = pep[j:]
                    break

            if j == len(pep) - 1 and pep[-1] in '.1234567890':  # till end
                mod[i] += sign * float(pep[1:])
                break
        else:
            seq += pep[0]
            pep = pep[1:]
            i = len(seq) - 1  # more realible

    return ''.join(seq), mod[:len(seq)], nmod


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
for i, a in enumerate(Alist):
    charMap[a] = i + 1

x_dim = oh_dim + 2
xshape = (max_in, x_dim)

# embed input item into a matrix


def embed(sp, mass_scale=max_mz, out=None, pep=None):
    if out is None:
        em = np.zeros((max_in, x_dim), dtype='float32')
    else:
        em = out

    if pep is None:
        pep = sp['pep']

    mod = sp['mod']

    if len(pep) > max_len:
        raise "input too long"

    em[len(pep)][21] = 1  # ending pos, next line with +1 to skip this
    for i in range(len(pep) + 1, max_in - 1):
        em[i][0] = 1  # padding first, meta column should not be affected

    meta = em[-1]
    meta[0] = fastmass(pep, ion_type='M', charge=1) / \
        mass_scale  # pos 0, and overwrtie above padding
    meta[sp['charge']] = 1  # pos 1 - 4
    meta[5 + sp['type']] = 1  # pos 5 - 8
    meta[-1] = sp['nce'] / 100.0 if 'nce' in sp else 0.25

    for i in range(len(pep)):
        em[i][charMap[pep[i]]] = 1  # 1 - 20
        em[i][-1] = mono[pep[i]] / mass_scale
        em[i][-2] = mod[i]

    return em


def f2(x): return "{0:.2f}".format(x)
def f4(x): return "{0:.4f}".format(x)

# function that transfer predictions into mgf format


def tomgf(sp, y):
    head = ("BEGIN IONS\n"
            f"TITLE={sp['title']}\n"
            f"PEPTIDE={sp['title']}\n"
            f"CHARGE={sp['charge']}+\n"
            f"PEPMASS={sp['mass']}\n")

    imz = np.arange(0, dim, dtype='int32') * precision + low  # more acurate

    mzs, its = sparse(imz, y)
    peaks = [f"{f2(mz)} {f4(it * 1000)}" for mz, it in zip(mzs, its)]

    return head + '\n'.join(peaks) + '\nEND IONS'


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    help='input file path', default='example.tsv')
parser.add_argument('--output', type=str,
                    help='output file path', default='example.mgf')
parser.add_argument('--model', type=str,
                    help='model file path', default='pm.h5')

args = parser.parse_args()

K.clear_session()

pm = k.models.load_model(args.model, compile=0)
# pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss='cosine')

# type: 0 unknown, 1 cid, 2 etd, 3 hcd
types = {'HCD': 3, 'ETD': 2}

# read inputs
inputs = []
for item in pd.read_csv(args.input, sep='\t').itertuples():
    if len(item.Peptide) > max_len:
        print("input", item.Peptide, 'exceed max length of', max_len, ", ignored")
        continue

    if item.Charge < 1 or item.Charge > 6:
        print("input", item.Peptide, 'has unspported charge state of',
              item.Charge, ", ignored")
        continue

    pep, mod, nterm_mod = getmod(item.Peptide)

    if nterm_mod != 0:
        print("input", item.Peptide, 'has N-term modification, ignored')
        continue

    if np.any(mod != 0) and set(mod) != set([0, 1]):
        print("Only Oxidation modification is supported, ignored input", item.Peptide)
        continue

    inputs.append({'pep': pep, 'mod': mod, 'charge': item.Charge, 'title': item.Peptide,
                   'nce': item.NCE, 'type': types[item.Type],
                   'mass': fastmass(pep, 'M', item.Charge, mod=mod)})


def input_generator(x, batch_size):
    while len(x) > batch_size:
        yield asnp32([embed(item) for item in x[:batch_size]])
        x = x[batch_size:]
    yield asnp32([embed(item) for item in x])


batch_size = 128
y = pm.predict(input_generator(inputs, batch_size), verbose=1,
               steps=int(math.ceil(len(inputs) / batch_size)))
y = np.square(y)

f = open(args.output, 'w+')
f.writelines("%s\n\n" % tomgf(sp, yi) for sp, yi in zip(inputs, y))
f.close()
