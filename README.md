# UnsupSeg
# Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation. ML course project.

general description, main contribution

# Datasets description
## TIMIT
BETTER TO ADD SOME GENERAL INFORMATION

Sourse: https://deepai.org/dataset/timit
TIMIT is English language dataset with size ~ 1.3Gb
Audio sampling frequency = 16kHz
It has standard train/test split (2 folders: TRAIN, TEST)
Preparing: code use standard train/test split and uses train data for train, randomly sampled 10% of train data for validation during training, test data for testing.
Files: 4158 - training, 462 - validation, 1680 - testing.
For project audio data (.wav) and phonemes data (.PHN) is required.
Each .PHN file contains start sample, end sample of phoneme and phoneme symbols.
Example: 9640 11240 sh, where 9640-start,11240-end,sh-phoneme
To process original dataset and extract (.wav) and (.PHN) files, processing script was written.
In the algorithm to process files, initial dataloader and code processing function (written by author) were used.

## Arabic

Source: http://en.arabicspeechcorpus.com/
For our research we used data from Arabic Speech Corpus dataset, which consists 1813 .wav audio files with spoken utterances with high quality and natural voice and 1813 .TextGrid text files with phoneme labels, time stamps of the boundaries where these occur in the audio files.
Initial sample rate of audio files was 48 kHz, and as our neural net uses only files with 16kHz we resampled it using bash script with using Sound eXchange (SoX) audio software.
For processing TextGrid files was used python library “textgrid”, that extracted all needed information from file. As time stamps in dataset in time domain we multiplied them to sample rate (16 000), to have sample stamps.
This dataset was used for testing procedure of the neural network that was trained on TIMIT dataset, so all 1813 files were used for testing. 
Size of dataset is ~500 MB




# How to use?
### Clone repository 
git clone https://github.com/Albly/UnsupSeg.git
cd UnsupSeg

### Setup environment
conda create --name unsup_seg --file requirements.txt

conda activate unsup_seg

### Data preperation

link to datasets

### Configuration

### Train / test / validation data structure
In the file config.yaml specify the directories for the location of datasets.
For each datased folder should look as follows:

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

### Train + Test

how to start train and test

### Test on a single audio file

how to start test on a single audio file
