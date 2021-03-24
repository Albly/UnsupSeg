import numpy as np
import h5py
from sklearn import preprocessing

'''
Description
    Preprocessing predicted and true data for classification. 

Actions:
    1) Load data from hdf5 files
    2) Delete outliers from predicted data
    3) Encode labels of phonemes
    4) Reshape vectors for convinient format
    5) Save arrays to .npz file

INPUT: 
    requres @phonemas_path, @boundaries_path, @scores_path hdf5, @realbounds_path @phoneme_symb_path files 

OUTPUT:
    .npz file with:
    X_train - predicted boundaries of phoneme
    X_test - real boundaries of phoneme
    y_test - encoded lables for X_test
    lables - labels of phonemes
'''

phonemas_path = 'Phonemas.hdf5'
boundaries_path = 'Boundaries.hdf5'
scores_path = 'Scores.hdf5'
realbounds_path = 'Realbounds.hdf5'
phoneme_symb_path = 'Phoneme_symb.hdf5'

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


# delete outliers using duration threshold
def delete_outlier(input_list,input_labels, min_th, max_th):
    print('Deleting outliers from train set')
    print('Input elements = ', len(input_list))
    z = []
    labels = []

    for item, label in zip(input_list, input_labels):
        if (item.shape[1] > min_th) and (item.shape[1] < max_th):
            z.append(item)
            labels.append(label)
    print('Out elements = ', len(z))
    print('Out labels = ', len(labels))
    return z, labels

def delete_outlier_test(input_list, min_th, max_th):
    print('Deleting outliers from test set')
    print('Input elements = ', len(input_list))
    z = []

    for item in input_list:
        if (item.shape[1] > min_th) and (item.shape[1] < max_th):
            z.append(item)
    print('Out elements = ', len(z))
    return z

def sum_along(z):
    new_z = []
    for z_el in z:
        a = np.sum(z_el, axis = 1).reshape((64,1))
        new_z.append(a)
    return new_z


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

#deleting outliers
z_test, phoneme_symbols = delete_outlier(z_spect_repr_slised_by_real, phoneme_symbols , 2, 25)
z_train =  delete_outlier_test(z_spect_repr_slised_by_predicted, 2, 25)

z_test = sum_along(z_test)
z_train = sum_along(z_train)

#encoding
labels = np.array(phoneme_symbols)
le = preprocessing.LabelEncoder()
y_test = le.fit_transform(labels)

#reshaping
X_train = np.array(z_train).reshape(-1,64)
X_test = np.array(z_test).reshape(-1,64)

#saving
np.savez('data_for_clf', X_train = X_train, X_test = X_test,  y_test = y_test, labels = labels )
print('data_for_clf.npz has generated')