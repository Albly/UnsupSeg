# UnsupSeg
# Self-Supervised Contrastive Learning for Unsupervised Phoneme Segmentation. ML course project.

general description, main contribution

# Datasets description

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
