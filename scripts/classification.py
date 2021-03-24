import numpy as np 
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

'''
Description
  Script for unsupervised classification of vowel/consonant phoneme (Binary)

Actions:
1) Takes prepared data from .npz file:
   X_train - predicted boundaries of phoneme
   X_test - real boundaries of phoneme
   y_test - encoded lables for X_test
   lables - labels of phonemes

2) Trains Kmeans and GaussianMixture algorithms on predicted boundaries of phonemes
3) Test models with X_test
4) Calculate number of vowel and consonant phonemes
5) Calculate metrics using true information about number of each group of phomonemes

INPUT: 
    requres @filepath for .npz files with data

OUTPUT:
    prints to terminal for each algorithm: 
    1) Specificity
    2  Sensitivity
    3) Balanced accuracy

'''

# filepath for .npz file with data
filepath = 'intermediate_data/data_for_clf.npz'


def get_cluster_data(cluster, preds, labels):
    '''
    Calculate number of unique phonemes and number of it's repetitions
    for defined cluster. Returns LABELS of predictions and their repetitions.

    INPUTS:
        @cluster - cluster value. Ex: 0,1,2...
        @preds - prediction codes from algorithms
        @labels - true labels in byte format (letters)
    
    OUTPUTS:
        @sorted_el_unique - sorted in descending order labels of phonemes
        @sorted_counts - sorted in descending order phoneme repetitions
    '''
    #indexes of prediction's elements that belong to requred cluster value
    idxs = np.argwhere(preds == cluster)
    #extract true labels for these indexes
    cluster_labels = labels[idxs]
    #find number of unique and repetions 
    el_unique, counts = np.unique(cluster_labels, return_counts=True)
    # sort by descending order results
    count_sort_ind = np.argsort(-counts)
    
    sorted_el_unique = el_unique[count_sort_ind]
    sorted_counts = counts[count_sort_ind]
    return sorted_el_unique, sorted_counts

def calculate_phonemes(labels, counts):
    '''
    Calculate number of vowels and consonants in @labels
    Also calculate number of unique vowels and consonants

    INPUTS:
        @labels - labels of phonemes
        @counts - number of label's repetitons 

    OUTPUTS:
        @vowels_count - number of vowels
        @vowels_unique_count - number of unique vowels
        @consonants_count -number of consonants
        @consonants_unique_count - number of  consonants

    '''

    vowels_sounds = ['a', 'e', 'i', 'o', 'u']

    vowels_unique_count = 0
    vowels_count = 0

    consonants_count = 0
    consonants_unique_count = 0

    for label, count in zip(labels, counts):
        label = label.decode('UTF-8')
        if label[0] in vowels_sounds:
            vowels_unique_count +=1
            vowels_count += count
        else:
            consonants_unique_count +=1
            consonants_count += count

    return vowels_count, vowels_unique_count, consonants_count, consonants_unique_count  


#loading file and data
file = np.load(filepath)
X_train = file['X_train']
X_test = file['X_test']

y_test = file['y_test']
labels = file['labels']


algorithms = [
                KMeans(n_clusters = 2, random_state = 0),
                GaussianMixture(n_components = 2)
             ]


for model in algorithms:
    model.fit(X_train)
    preds = model.predict(X_test)

    #get data from first cluster 
    unique_elemenst, counts = get_cluster_data(0, preds , labels)
    a1,_,b1,_ = calculate_phonemes(unique_elemenst, counts) 

    #get data from second cluster 
    unique_elemenst, counts = get_cluster_data(1, preds , labels)
    a2,_,b2,_ = calculate_phonemes(unique_elemenst, counts)

    #defienes metrics
    if a1 > b1:
        tp,fp,fn,tn = a1,b1,a2,b2
    else:
        tp,fp,fn,tn = a2,b2,a1,b1

    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)

    balanced_accuracy = (sensitivity+specificity)/2

    print('==========================================')
    print('Results for :       '+ type(model).__name__)
    print('Specificity :       '+ str(np.round(specificity,3)))
    print('Sensitivity :       '+ str(np.round(sensitivity,3)))
    print('Balanced accuracy : '+ str(np.round(balanced_accuracy,3)))

