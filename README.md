[Experiments_ntbk]:https://github.com/Albly/UnsupSeg/blob/master/Experiments_results.ipynb 
[Report_ntbk]:https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb
[Original_paper_GH]:https://github.com/felixkreuk/UnsupSeg
[Original_paper]:https://arxiv.org/abs/2007.13465

# Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation. ML course project, Skoltech.
## General Description
This repository is an unofficial `Python` replication and implementation of the paper ["Self-Supervised Con-trastive Learning for Unsupervised Phoneme Segmentation(Kreuk et al., 2020)](https://arxiv.org/abs/2007.13465). 

The original model's performance was tested on an out-of-domain test set. Furthermore, modifications and improvements to the original loss function and model architecture were implemented which led to comparable results and, in some cases, better than those from the source paper.

Phonemes clusterization was also implemented.

The files with the code were taken from the author's GitHub and some of them were changed for our tasks.

## Main Contributions
The main contributions include:
1. We replicated the originally proposed model on the TIMIT dataset and profiled the performance.(see `Metrics for Task 1` section [here](https://github.com/Albly/UnsupSeg/blob/master/Experiments_results.ipynb ) and `Main for Training` section [here](https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb))
2. We analysed the TIMIT-trained model's performance on 'Arabic Speech Corpus', which is an out-of-domain dataset (see `Metrics for Task 2` section [here](https://github.com/Albly/UnsupSeg/blob/master/Experiments_results.ipynb )).
3. We improved the performance of the model by experimenting with different loss-functions (see `Metrics for Task 3` [here](https://github.com/Albly/UnsupSeg/blob/master/Experiments_results.ipynb )).
4. We show that the model's performance is improved by applying a windowed Fast Fourier Transform over the audio samples (see lines [34 - 37 here](https://github.com/Albly/UnsupSeg/blob/master/next_frame_classifier.py)). 
5. We performed clusterization of the phonemes on the TIMIT dataset (see `Saving test results: phonemas, boundaries...` [here](https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb)).

# Within the framework of this work we had the following tasks:
1. Replicate the results on one of the datasets.
2. Analyze performance on out of the domain test set.
3. Propose and implement improvements for the loss function
4. Try to find a number of unique phonemes
5. Try to extend to unsupervised phoneme classification

## Files and folders description
Experiments_results.ipynb - consists of experiments results (train/test metrics, their averaging and plotting)
You don't have to execute cells in this file. Just obtain the results.

[`Report.ipynb`](https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb) - consists of subblocks for different tasks:
1. Dataloaders testing (for TIMIT, ArabicSpeech datasets).
2. Saving real bounds and phoneme labels into hdf5 file (for TIMIT).
3. Test pre-trained model on a single audio file from the test data set.
5. Train model on a train data set, test on a pre-trained model.
7. Test data saving (for convenient following processing, HDF5 files):
    - from the network: spectral representations of audio, scores, predicted boundaries of phonemes,
    - from test data set: real boundaries of phonemes, phonemes characters.
8. Data reading from written files.
9. Plotting example: real boundaries of phonemes, spectral representations of audio, scores, predicted boundaries of phonemes.
10. TIMIT data set parser.
11. Threshold-based algorithm for outliers detecting and comparison of real and predicted boundaries distributions.

[`config.yaml`](https://github.com/Albly/UnsupSeg/blob/master/config.yaml) - contains model hyperparameters and other parameters like paths to train/test folders etc.

[`dataloader.py`](https://github.com/Albly/UnsupSeg/blob/master/dataloader.py) - is responsible for data reading from datasets in the appropriate format.

[`predict.py`](https://github.com/Albly/UnsupSeg/blob/master/predict.py) - to make a prediction on a single audio file and to save real boundaries and scores into hdf5 files.

[`solver.py`](https://github.com/Albly/UnsupSeg/blob/master/solver.py) - solver for the model and its functions (forward, optimizer, building the model etc.).

[`utils.py`](https://github.com/Albly/UnsupSeg/blob/master/utils.py) - some functions used in other scripts (metrics evaluation, peak detection etc.).

[`scripts folder`](https://github.com/Albly/UnsupSeg/tree/master/scripts) - scripts with model training, and data preprocessing.

# Datasets description
## TIMIT
 [TIMIT](https://deepai.org/dataset/timit) is an English language dataset containing recording of 630 speakers with 8 differents dialects (DR1, DR2, ..., DR8) of American English. In the recordings, each individual has 10 sentences that are rich in phonetics, making it a great dataset for phoneme segmentation. It has an audio sampling frequency = 16kHz and has standardized train/test split (2 folders: TRAIN, TEST). 

Our code uses this standard train/test split provided in the TIMIT dataset. All train samples from all dialect regions (DR1, DR2,...,DR8) are placed in one folder. A randomly sampled 10% set of this combined train data is used for validation during training resulting in: 
- 4158 files for training, 
- 462 files for  validation and
- 1680 for testing.

For project audio data (.wav) and phonemes data (.PHN) is required.
Each .PHN file contains start sample, end sample of phoneme and phoneme symbols.
Example: 9640 11240 sh, where 9640-start,11240-end,sh-phoneme
To process original dataset and extract (.wav) and (.PHN) files into their respective train-test folders, a processing script was written (see `big_timit_parser()` in `Report.ipynb`).
In the algorithm to process files, initial dataloader and code processing function [written by the original paper authors](https://arxiv.org/abs/2007.13465) was used as reference.

## Arabic Speech Corpus

For our research, we used data from [Arabic Speech Corpus dataset](http://en.arabicspeechcorpus.com/), which consists of 1813 .wav audio files with spoken utterances with high quality and natural voice and 1813 .TextGrid text files with phoneme labels, timestamps of the boundaries where these occur in the audio files.
The initial sample rate of audio files was 48 kHz, and as our neural net uses only files with 16kHz we resampled it using bash script with using Sound eXchange (SoX) audio software.
For processing TextGrid files was used python library “textgrid”, which extracted all needed information from a file. As timestamps in a dataset in the time domain, we multiplied them to sample rate (16 000), to have sample stamps.
This dataset was used for the testing procedure of the neural network that was trained on the TIMIT dataset, so all 1813 files were used for testing. 
The size of dataset is ~500 MB

# Usage
### Prerequisites
- Linux or macOS
- conda
### Clone repository 
```sh
git clone https://github.com/Albly/UnsupSeg.git
cd UnsupSeg
```
# Download data folder
https://drive.google.com/file/d/1EicsBqFk_qIWg3v8iDAr1yAO1w3CmAek/view?usp=sharing

This folder contains 2 data sets and hdf5 files written on the TIMIT data set

### Setup environment
```
conda create --name unsup_seg --file requirements.txt
conda activate unsup_seg
conda install -c conda-forge jupyterlab
```
### Data preperation (better to use prepared folder 'data')
This example shows how to prepare the TIMIT dataset for training. It can be adopted for other datasets as well.
1. Download the TIMIT dataset from https://data.deepai.org/timit.zip
2. Extract the data into `/other/timit_big/data`
3. Open `Report.ipynb`  from the working directory
    ```
    jupyter notebook Report.ipynb
    ```
4. Run the cell under `TIMIT parser` to parse the data into train and test. The parsed data will be stored in `other/timit_big_parsed` in the format described in the data structure section described. 
5. Copy the created train and test folders from `other/timit_big_parsed` into your prefered data location according to your `timit_path` in `config.yaml`
```
cp -avr /other/timit_big_parsed /path/to/timit/timit
```

### Configuration
Before using the code, configure the paths to the datasets eg timit in `config.yaml`. The directory should contain train,test and val subfolders (see data-structure description below). Configure other parameters relevant for training, validation and testing in `config.yaml`
```
...
# DATA
timit_path /path/to/timit/timit
...
```
### Train / test / validation data structure
In the file config.yaml specify the directories for the location of datasets.
For each datased folder should look as follows:
```
UndupSeg
│
│
data + intermediate_data, utils.py, solver.py, predict.py,
│      dataloader.py, config.yaml, Report.ipynb,Experiments_results.ipynb
│
datasets
│
│
timit (same for Arabic Speech Corpus: arabic)
│
│
 └───val
  │   │   X.wav (or X.WAW)
  │   └─  X.phn (or X.TextGrid)
  │
  └───test
  │   │   Y.wav
  │   └─  Y.phn
  │
  └───train
      │   Z.wav
      └─  Z.phn
```
# How to reproduce results?
1. Implement pre requirements: setup environment, prepare the data, organise folder structure
2. Open `Report.ipynb`
3. First of all - to be sure that data loaders are working and read the dataset in appropriate format (the result of cell execution should be the same)
4. The second find `Test on single audio` in [`Report.ipynb`](https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb) and try running the code under it and obtain results
5. The third find `Main for training and testing` in [`Report.ipynb`](https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb)  and run the corresponding cells. Don't forget to edit config.yaml to set up hyperparameters and to set the mode (train/test). For the train ckpt: - relative path to the model. For the test ckpt: null. Also, choose the dataset folder (data) from the datasets you have and correct paths to them.
6. Obtain the training/testing procedure.
7. To obtain examples of data from the core of the network you can use the code under `Plot example of data: spectral reprezentation, score and boundaries` in [`Report.ipynb`](https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb)
8. To implement outliers detecting based on duration threshold, run cells under `Outliers detecting (by phoneme duration)` in [`Report.ipynb`](https://github.com/Albly/UnsupSeg/blob/master/Report.ipynb)

## More more detailed information is in the descriptions and comments of each file
