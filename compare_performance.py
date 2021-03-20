import argparse
import math
import numpy as np
from pyteomics import mgf, mass


def norm(x): return np.linalg.norm(x)


def cosine(u, v): return np.dot(u, v) / (norm(u) * norm(v) + 1e-16)


DIMENSION = 20000
BIN_SIZE = 0.1


def spectrum2vector(mz_list, itensity_list, mass, bin_size, charge):
    itensity_list = itensity_list / np.max(itensity_list)

    vector = np.zeros(DIMENSION, dtype='float32')

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


# ratio constants for NCE
cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}


def parse_spectra(sps):
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
    return codes


parser = argparse.ArgumentParser()
parser.add_argument('--real', type=str,
                    help='Real MGF file path', default='hcd_testingset.mgf')
parser.add_argument('--pred', type=str,
                    help='predicted MGF file path', default='example_prediction.mgf')

args = parser.parse_args()

print('Reading', args.real)
real_vectors = [spectrum2vector(sp['mz'], sp['it'], sp['mass'], BIN_SIZE,
                               sp['charge']) for sp in readmgf(args.real)]

print('Reading', args.pred)
pred_vectors = [spectrum2vector(sp['mz'], sp['it'], sp['mass'], BIN_SIZE,
                               sp['charge']) for sp in readmgf(args.pred)]

similarites = [cosine(sp1, sp2)
               for sp1, sp2 in zip(real_vectors, pred_vectors)]

print('Average Cosine similarites:', np.mean(similarites))
