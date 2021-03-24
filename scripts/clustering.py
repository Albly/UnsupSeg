from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, SpectralClustering,AgglomerativeClustering
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm import tqdm

'''
Description
    Script for training KMeans, GaussianMixture, AgglomerativeClustering, Birch models
    for different number of clusters.
    For each model evaluates silhouette_score
    Save arrays with number of clusters and silhouette_score to .npz file

Actions:
    1) load phonemes, boundaries, scores, from files 
    2) Splits data to 3 bathes (otherwise AgglomerativeClustering, Birch cause out of memory)
    3) Trains KMeans, GaussianMixture, AgglomerativeClustering, Birch models with different number of clusters
    4) For each model evaluate silhouette_score
    5) Save number of clusters and silhouette_score to .npz file

INPUT: 
    requres @phonemas_path, @boundaries_path, @scores_path hdf5 files 

OUTPUT:
    Save array as .npz file with array for each algorithm with number of clusters and silhouette_score

'''

phonemas_path = 'data/Phonemas.hdf5'
boundaries_path = 'data/Boundaries.hdf5'
scores_path = 'data/Scores.hdf5'
end_folder = 'intermediate_data/'

def read_data():
    '''
    Reads data from hdf5 files 
    '''

    phonemas = []
    boundaries = []
    scores = []

    with h5py.File(phonemas_path, 'r') as f:
        print('phonemas file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            phonemas.append(f['phoneme_'+str(i)][:])

    with h5py.File(boundaries_path, 'r') as f:
        print('boundaries file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            boundaries.append(f['bounds_'+str(i)][:])

    with h5py.File(scores_path, 'r') as f:
        print('scores file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            scores.append(f['scores_'+str(i)][:])
            
    # phonema slicing
    phonema_sliced = []
    for spectrum_idx in range(len(phonemas)):
        
        for bound_idx in range(len(boundaries[spectrum_idx])-1):
            phonema = phonemas[spectrum_idx][:,boundaries[spectrum_idx][bound_idx]:boundaries[spectrum_idx][bound_idx+1]].sum(axis = 1).reshape(64,1)
            phonema_sliced.append(phonema)
        
    print()
    print('phonemas ',len(phonemas))
    print('boundaries ',len(boundaries))
    print('scores ',len(scores))
    print()
    print('Phonemas detected:',len(phonema_sliced))
    return phonemas, boundaries, scores, phonema_sliced


def train_with_grid(grid,X,i):
    '''
    Function for training algorithms 
    INPUTS: 
        @grid - dict with grid parameters
        @X - input data for training
        @i - number of batch
    '''
    sil_scores = []
    n_clust = []

    for param in tqdm(ParameterGrid(grid)):

        if param['method'] == 'KMeans':
            model = KMeans(n_clusters = param['n_clusters'], random_state = random_seed, n_jobs=-1)

        elif param['method'] == 'GaussianMixture':
            model = GaussianMixture(n_components = param['n_clusters'], random_state = random_seed)

        elif param['method'] == 'Birch':
            model = Birch(n_clusters = param['n_clusters'] ,copy = False)
        
        elif param['method'] == 'AgglomerativeClustering':
            model = AgglomerativeClustering(n_clusters = param['n_clusters'])

        preds = model.fit_predict(X)
        silhouette = silhouette_score(X, preds)
        sil_scores.append(silhouette)
        n_clust.append(param['n_clusters'])

    np.savez(end_folder+param['method']+str(i), np.array(sil_scores), np.array(n_clust))

random_seed = 0

phonemas, boundaries, scores, phonema_sliced = read_data()
X = np.array(phonema_sliced).reshape(len(phonema_sliced),64)

X_splits = np.array_split(X, 3) 

n_clusters = [2,3,4,5,6,10,15,20,30,40,45,50,55,60,65]

grids = [
       {'method': ['Birch'], 'n_clusters':n_clusters},
       {'method': ['AgglomerativeClustering'], 'n_clusters':n_clusters},
       {'method': ['KMeans'], 'n_clusters':n_clusters },
       {'method': ['GaussianMixture'], 'n_clusters':n_clusters  }
       ]

# training procedure
for i,x in enumerate(X_splits):
    for grid in grids:
        train_with_grid(grid,x,i)