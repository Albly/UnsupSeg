import umap.umap_ as umap
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm

'''
Description
    Script for collecting data for visualization using dimentionality reduction algorithms
    UMAP and t-SNE.
    For visualization true specreal representations.

Actions:
    1) load phonemes, boundaries, scores, from files 
    2) Reduce dims for specreal representations with UMAP and t-SNE
    3) Save embedings and true labels to .npz files


INPUT: 
    requres @phonemas_path, @boundaries_path, @scores_path hdf5, @realbounds_path @phoneme_symb_path files 
    
OUTPUT:
    .npz files with embedings and true labels for both algorithms 
'''

phonemas_path = 'data/Phonemas.hdf5'
boundaries_path = 'data/Boundaries.hdf5'
scores_path = 'data/Scores.hdf5'
realbounds_path = 'data/Realbounds.hdf5'
phoneme_symb_path = 'data/Phoneme_symb.hdf5'

end_folder = 'intermediate_data/'

# slice phonemes using boundaries as input. Don't consider left and right sides as phonemes
def slice_phonemaes(phns, bnds):
    phonema_sliced = []
    phonema_sliced_sum = []
    for spectrum_idx in range(len(phns)):
        for bound_idx in range(len(bnds[spectrum_idx])):
            if bound_idx == 0:
                None
            else:
                # print(bnds[spectrum_idx][bound_idx-1],bnds[spectrum_idx][bound_idx])
                phonema = phns[spectrum_idx][:,bnds[spectrum_idx][bound_idx-1]:bnds[spectrum_idx][bound_idx]]
                phonema_sliced.append(phonema) # full representation of phoneme
                phonema_sliced_sum.append(phonema.sum(axis = 1).reshape(64,1)) # short representation of phoneme
    return phonema_sliced, phonema_sliced_sum

# extract phoneme symbols and z_spect representations slised by real boundaries
def get_phoneme_symbols(phn_sym):
    phn_sym = []
    for item in phn_symbols:
        phn_sym.extend(item[1:-1])
    print('Real phoneme symbols = ', len(phn_sym))
    return phn_sym

def read_data():
    import h5py
    z_spectral_representations = [] # set of z spectral representations for each test file
    boundaries_predicted = [] # predicted boundaries for test data
    bounds_real = [] # real boundaries for test data
    scores = [] # scores for test data
    phn_symbols = [] # phoneme symbols for test data

    with h5py.File(phonemas_path, 'r') as f:
        print('phonemas file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            z_spectral_representations.append(f['phoneme_'+str(i)][:])

    with h5py.File(boundaries_path, 'r') as f:
        print('boundaries file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            boundaries_predicted.append(f['bounds_'+str(i)][:])

    with h5py.File(scores_path, 'r') as f:
        print('scores file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            scores.append(f['scores_'+str(i)][:])
            
    with h5py.File(realbounds_path, 'r') as f:
        print('realbounds file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            bounds_real.append(f['real_bound_'+str(i)][:])
    bounds_real = [i.astype('int') for i in bounds_real]
        
    with h5py.File(phoneme_symb_path, 'r') as f:
        print('Phoneme_symb file ',len(f.keys()))
        for i in range(int(len(f.keys()))):
            phn_symbols.append(f['phn_symb_'+str(i)][:])


    print()
    print('z_spectral_representations ',len(z_spectral_representations))
    print('boundaries_predicted ',len(boundaries_predicted))
    print('scores ',len(scores))
    print('bounds_real ',len(bounds_real))
    print()

    return z_spectral_representations, phn_symbols, scores, boundaries_predicted, bounds_real


def UMAP_reduction(X, target):

    '''
    UMAP dim reduction model

    INPUTS: 
        @X - input vector
        @target - encoded label (for colors)  
    '''

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)

    print("Umap successfully trained")
    path = end_folder+'umap_'+str(2)
    np.savez(path, embedding, target)
    print(path+'.npz',' successfully generated')


def tSNE_reduction(X, target):

    '''
    tSNE dim reduction model

    INPUTS: 
        @X - input vector
        @target - encoded label (for colors)  
    '''

    tsne = TSNE(random_state=33, n_components=2)
    X_tsne = tsne.fit_transform(X)

    print("TSNE successfully trained")

    path = end_folder+'tsne_'+str(2)
    np.savez(path, X_tsne, target)
    print(path+'.npz',' successfully generated')


z_spectral_representations, phn_symbols, scores, boundaries_predicted, bounds_real = read_data()

# phonema slicing:
# by predicted bounds
z_spect_repr_slised_by_predicted, z_spect_repr_slised_sum_by_predicted = slice_phonemaes(z_spectral_representations,boundaries_predicted)
print('Predicted phonemes in total:',len(z_spect_repr_slised_by_predicted))

# by real bounds
z_spect_repr_slised_by_real, z_spect_repr_slised_sum_by_real = slice_phonemaes(z_spectral_representations,bounds_real)
print('Real phonemes in total: ', len(z_spect_repr_slised_by_real))

# and get phoneme symbols
phoneme_symbols = get_phoneme_symbols(phn_symbols)


target = np.array(phoneme_symbols)
#encoding labels
le = preprocessing.LabelEncoder()
target = le.fit_transform(target)

X = np.array(z_spect_repr_slised_sum_by_real).reshape(-1,64)

print('Trainting algorithms...')
UMAP_reduction(X, target)
tSNE_reduction(X, target)

