# PredFull

__Visit [http://predfull.com/](http://predfull.com/) to try online prediction__

Model for [`Full-Spectrum Prediction of Peptides Tandem Mass Spectra using Deep Neural Network`](https://pubs.acs.org/doi/10.1021/acs.analchem.9b04867)

Kaiyuan Liu, Sujun Li, Lei Wang, Yuzhen Ye, Haixu Tang

## Method

Based on the structure of the residual convolutional networks. Current precision: 0.1 Th.

![model](imgs/model.png)

## How to use

### Required Packages

Recommend to install dependency via [Anaconda](https://www.anaconda.com/distribution/)

* Keras >= 2.2.4
* Tensorflow >= 1.12
* Pandas >= 0.20
* pyteomics
* lxml

### Input format

The required input format is TSV, with following columns:

Peptide | Charge | Type | NCE
------- | ------ | ---- | ---
AAAAAAAAAVSR | 2 | HCD | 25
AAGAAESEEDFLR | 2 | HCD | 25
AAPAPTASSTININTSTSK | 2 | HCD | 25

Apparently, 'Peptide' and 'Charge' columns mean what it says. The 'Type' must be HCD or ETD (in uppercase). NCE means normalized collision energy, set to 25 or 30 if you don't care. Check `example.tsv` for examples.

### Usage

Simply run:

`python predfull.py --input .\example.tsv --weights pm.hdf5 --output .\example.mgf`

The output file is in MGF format

* --input : the input file
* --weights : the weight file
* --output : the output path

## Prediction Examples

![example 1](imgs/hcd2.png)

![example 2](imgs/hcd1.png)
