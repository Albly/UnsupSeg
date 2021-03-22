# Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation. ML course project, Skoltech.
## General Description
This repository is an unofficial `Python` replication and implementation of the paper ["Self-Supervised Con-trastive Learning for Unsupervised Phoneme Seg-mentation’(Kreuk et al., 2020)](https://arxiv.org/abs/2007.13465). 

The original model's performance was tested on an out-of-domain test set. Furthermore, modifications and improvements to the original loss function and model architecture were implemented which led to results that are comparable and, in some cases, better that those from the source paper.

Phonemes clusterization was also implemented.

The files with the code were taken from the author's github and some of them were changed for our tasks.

## Main Contributions
The main contributions include:
1. We replicated the original proposed model on the TIMIT dataset and profiled the performace.
2. We analysed the TIMIT-trained model's performance on 'Arabic Speech Corpus', which is an out-of-domain dataset. 
3. We improved the performance of the model by experimenting with different loss-functions.
4. We show that the model's performance is improved by applying a windowed Fast Fourier Transform over the audio samples. 
5. We performed unique clusterization of the phonemes on the TIMIT dataset.

## Files description
Experiments_results.ipynb - consists of experiments results (train/test metrics, their averaging and plotting)
Report.ipynb - consists of subblocks for different tasks:
```
    1.Test pretrained model on a single audio file from test data set
    2.Train model on a train data set
    3.Dataloaders testing (for TIMIT, ArabicSpeech and Buckeye datasets)
    4.Test data saving (for conveinient following processing, HDF5 files):
        4.1.from network: spectral representationsof audio, scores, predicted boundaries of phonemes
        4.2.from test data set: real boundaries of phonemes, phonemes characters
    5.Data reading from written files
    6.Plotting example:real boundaries of phonemes,spectral representationsof audio, scores, predicted boundaries of phonemes
    7.TIMIT data set parser
    8.Algorithm for outliers detecting
```
# Datasets description
## TIMIT

[TIMIT](https://deepai.org/dataset/timit) is an English language dataset with size ~ 1.3Gb
Audio sampling frequency = 16kHz
It has standard train/test split (2 folders: TRAIN, TEST)
`Data Preparation:` the code uses standard train/test split provided in the TIMIT dataset. All train samples from all dialect regions (DR1, DR2,...,DR8) are placed in one folder. A Randomly sampled 10% set of this combined train data is used for validation during training. 
Files: 4158 - training, 462 - validation, 1680 - testing.
For project audio data (.wav) and phonemes data (.PHN) is required.
Each .PHN file contains start sample, end sample of phoneme and phoneme symbols.

Example: 9640 11240 sh, where 9640 - start,11240 - end,sh - phoneme

To process original dataset and extract (.wav) and (.PHN) files into their respective train-test folders, a processing script was written (see `big_timit_parser()` in `Report.ipynb`).
In the algorithm to process files, initial dataloader and code processing function [written by the original paper authors](https://github.com/felixkreuk/UnsupSeg) was used.

## Arabic Speech Corpus

For our research we used data from [Arabic Speech Corpus dataset](http://en.arabicspeechcorpus.com/), which consists 1813 .wav audio files with spoken utterances with high quality and natural voice and 1813 .TextGrid text files with phoneme labels, time stamps of the boundaries where these occur in the audio files.
Initial sample rate of audio files was 48 kHz, and as our neural net uses only files with 16kHz we resampled it using bash script with using Sound eXchange (SoX) audio software.
For processing TextGrid files was used python library “textgrid”, that extracted all needed information from file. As time stamps in dataset in time domain we multiplied them to sample rate (16 000), to have sample stamps.
This dataset was used for testing procedure of the neural network that was trained on TIMIT dataset, so all 1813 files were used for testing. 
Size of dataset is ~500 MB

# Usage
### Prerequisites
- Linux or macOS
- conda
### Clone repository 
```sh
git clone https://github.com/Albly/UnsupSeg.git
cd UnsupSeg
```
### Setup environment
```
conda create --name unsup_seg --file requirements.txt
conda activate unsup_seg
conda install -c conda-forge jupyterlab
```
### Data preperation
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
timit_directory
│
 └───val
  │   │   X.wav
  │   └─  X.phn
  │
  └───test
  │   │   Y.wav
  │   └─  Y.phn
  │
  └───train
      │   Z.wav
      └─  Z.phn
```
# DATA
timit_path /path/to/timit/timit
### Train + Test

how to start train and test

### Test on a single audio file

how to start test on a single audio file
