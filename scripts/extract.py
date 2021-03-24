import numpy as np
import matplotlib.pyplot as plt

'''
Description
    Script for plotting graphs for clustering algorithms
    with number of clusters and Silhouette scores 

Actions:
    1) Search .npz files in @folder path for Birch, GaussianMixture, KMeans,AgglomerativeClustering
    2) Read data for each algorithm and batch
    3) Average results over all batches
    4) plot curve n_clusters vs Silhouette

INPUT: 
    requres: @folder path with .npz files

OUTPUT:
    Plot with curves n_clusters vs Silhouette 
'''

folder = ''

def plt_averaged(algorithms,batches):
    results = []

    for algorithm in algorithms:
   
        y_buffer = []
        x_buffer = []
        result = {}

        for batch in batches:
            #path to file
            path = folder+algorithm + batch + '.npz'
            #load file
            file = np.load(path)
            #extract data
            y = file['arr_0']
            x = file['arr_1']
            #add data to buffer
            y_buffer.append(y)
            x_buffer = x

        # mean over all batches
        y_avg = np.mean(np.array(y_buffer), axis=0)
        
        #save result to dict
        result['algorithm'] = algorithm
        result['y_avg'] = y_avg
        result['x_val'] = x_buffer
        results.append(result)
        
    return results

# names of algorithms
algorithms = ['Birch', 'GaussianMixture', 'KMeans', 'AgglomerativeClustering']
# number of batch
batches = ['0','1','2']

#get results
results = plt_averaged(algorithms, batches)

#plotting
plt.figure(figsize=(10,5))
for result in results:
    x = result['x_val']
    y = result['y_avg']
    plt.plot( x, y, label = result['algorithm'])

plt.xlabel("Number of clusters")
plt.ylabel("Silhouette")
plt.legend()
plt.xticks(np.arange(2,66,4))
plt.grid(alpha =0.5)
plt.show()