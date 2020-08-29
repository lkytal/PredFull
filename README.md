# PredFull

__Visit [http://predfull.com/](http://predfull.com/) to try online prediction__

This work was published on Analytical Chemistry: [`Full-Spectrum Prediction of Peptides Tandem Mass Spectra using Deep Neural Network`](https://pubs.acs.org/doi/10.1021/acs.analchem.9b04867)

Kaiyuan Liu, Sujun Li, Lei Wang, Yuzhen Ye, Haixu Tang

## Update History

* 2020.08.22: Fixed performance issues
* 2020.05.25: Support predicting non-tryptic peptides
* 2019.09.01: First version


## Method

Based on the structure of the residual convolutional networks. Current precision (bin size): 0.1 Th.

![model](imgs/model.png)

## How to use

### Important Notes

* This model support only UNMODIFIED peptides (for now, at least)
* This model assume a FIXED carbamidomethyl on C
* The length of input peptides are limited to =< 30
* The prediction will NOT output peaks with M/z > 2000
* Predicted peaks that are weaker than STRONGEST_PEAK / 1000 are regarded as noises and ignored from the output.

### Required Packages

Recommend to install dependency via [Anaconda](https://www.anaconda.com/distribution/)

__The Tensorflow has to be 2.30 or newer! A compatibility bug in Tensorflow made version before 2.3.0 can't load the model correctly. We'll release a new model once they solve this.__

* Python >= 3.7
* Tensorflow >= 2.3.0
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

Apparently, 'Peptide' and 'Charge' columns mean what it says. The 'Type' must be HCD or ETD (in uppercase). NCE means normalized collision energy, set to 25 as default (Notice: the released model newer trained on samples with NCE > 50). Check `example.tsv` for examples.

### Usage

Simply run:

`python predfull.py --input example.tsv --model pm.h5 --output example.mgf`

The output file is in MGF format

* --input : the input file
* --output : the output path
* --model : the pretrained model

## Prediction Examples

![example 1](imgs/hcd2.png)

![example 2](imgs/hcd1.png)

## Performance Evaluation

We provide sample data and codes for you to evaluate the prediction performance. The `hcd_testingset.mgf` file contains ground truth spectra (extracted from NIST) that corresponding to items in `example.tsv`,  while `example.mgf` are pre-runned prediction results of `example.tsv`.

To evaluate the similarity, run:

`python compare_performance.py --real hcd_testingset.mgf --pred example.mgf`

* --real : the ground truth file
* --pred : the predcition file

You sholud get around $0.8027$ average similarities on this given data.

__Make sure that items in `example.tsv` and `hcd_testingset.mgf` are of same order! Don't permute items or add/delete items unless you will align them by yourself.__

## How to build & train the model

For who interested in reproduce this model, here we provide `build_model_example.py` of example codes to build the model. More details of how to train this model will release later.
