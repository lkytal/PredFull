# PredFull

__Visit [http://predfull.com/](http://predfull.com/) to try online prediction__

> This work was published on Analytical Chemistry: [`Full-Spectrum Prediction of Peptides Tandem Mass Spectra using Deep Neural Network`](https://pubs.acs.org/doi/10.1021/acs.analchem.9b04867)
>
> Kaiyuan Liu, Sujun Li, Lei Wang, Yuzhen Ye, Haixu Tang

The first model for predicting complete tandem mass spectra from peptides sequences, using a deep CNN neural network trained on over 2 million experimental spectra.

## Update History

* 2021.05.18: Support predicting peptides with oxidized methionine.
* 2021.01.01: Update example results.
* 2020.08.22: Fixed performance issues.
* 2020.05.25: Support predicting non-tryptic peptides.
* 2019.09.01: First version.

## Method

Based on the structure of the residual convolutional networks. Current precision (bin size): 0.1 Th.

![model](imgs/model.png)

## How to use

__Expect clone this project, you should download `pm.h5` from [google drive](https://drive.google.com/drive/folders/1KQqQGTSY5y2w3cQV1zKzuGbhgThCS9vn?usp=sharing) and place it into this folder.__

### Important Notes

* The only modification (PTM) supported is **oxidation on Methionine**, otherwise only UNMODIFIED peptides are allowed. To indicate an oxidized methionine, use the format "M(O)".
* This model assumes a __FIXED__ carbamidomethyl on C
* The length of input peptides are limited to =< 30
* The prediction will NOT output peaks with M/z > 2000
* Predicted peaks that are weaker than STRONGEST_PEAK / 1000 are regarded as noises thus will be omitted from the final output.

### Required Packages

Recommend to install dependency via [Anaconda](https://www.anaconda.com/distribution/)

* Python >= 3.7
* Tensorflow >= 2.3.0
* Pandas >= 0.20
* pyteomics
* lxml

__The Tensorflow has to be 2.30 or newer! A compatibility bug in Tensorflow made version before 2.3.0 can't load the model correctly. We'll release a new model once the Tensorflow team solve this.__

### Input format

The required input format is TSV, with the following columns:

Peptide | Charge | Type | NCE
------- | ------ | ---- | ---
AAAAAAAAAVSR | 2 | HCD | 25
AAGAAESEEDFLR | 2 | HCD | 25
AAPAPTASSTININTSTSK | 2 | HCD | 25
AAPAPM(O)NTSTSK | 2 | HCD | 25

Apparently, 'Peptide' and 'Charge' columns mean what it says. The 'Type' must be HCD or ETD (in uppercase). NCE means normalized collision energy, set to 25 as default. Note that in the above examples the last peptide has an oxidized methionine, and it's the only modification supported now. Check `example.tsv` for examples.

### Usage

Simply run:

`python predfull.py --input example.tsv --model pm.h5 --output example_prediction.mgf`

The output file is in MGF format

* --input: the input file
* --output: the output path
* --model: the pretrained model

## Prediction Examples

__Note that intensities are shown by square rooted values__

![example 1](imgs/hcd2.png)

![example 2](imgs/hcd1.png)

## Performance Evaluation

We provide sample data on [google drive](https://drive.google.com/drive/folders/1KQqQGTSY5y2w3cQV1zKzuGbhgThCS9vn?usp=sharing) and codes for you to evaluate the prediction performance. The `hcd_testingset.mgf` file on google drive contains ground truth spectra (randomly sampled from [NIST Human Synthetic Peptide Spectral Library](https://chemdata.nist.gov/dokuwiki/doku.php?id=peptidew:lib:kustersynselected20170530)) that corresponding to items in `example.tsv`, while the `example_prediction.mgf` file contains pre-run predictions.

To evaluate the similarity, first download groud truth reference file `hcd_testingset.mgf` from [google drive](https://drive.google.com/drive/folders/1KQqQGTSY5y2w3cQV1zKzuGbhgThCS9vn?usp=sharing), then run:

`python compare_performance.py --real hcd_testingset.mgf --pred example_prediction.mgf`

* --real: the ground truth file
* --pred: the prediction file

You should get around ~0.789 average similarities using these two pre-given MGF files.

__Make sure that items in `example.tsv` and `hcd_testingset.mgf` are of the same order! Don't permute items or add/delete items unless you will align them by yourself.__

## How to build & train the model

For those who are interested in reproducing this model, here we provide `train_model.py` of example codes to build and train the model.
