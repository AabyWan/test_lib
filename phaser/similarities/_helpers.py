import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from itertools import combinations

def find_inter_samplesize(num_images) -> int:
    for n in range(0,num_images):
        if (n * (n-1))/2 > num_images : 
            return n
    return 0

# DISTANCE COMPUTATION
class IntraDistance():
    def __init__(self, le_t, le_m, le_a, set_class=0):
        """_summary_

        Parameters
        ----------
        le_t : sklearn.preprocessing.LabelEncoder
            _description_
        
        le_m : sklearn.preprocessing.LabelEncoder
            _description_
        
        le_a : sklearn.preprocessing.LabelEncoder
            _description_

        set_class : int, optional
            _description_, by default 0
        """
        self.le_t = le_t
        self.le_m = le_m
        self.le_a = le_a
        self.set_class = set_class

    def intradistance(self, x, algorithm, metric):
        # store the first hash and reshape into 2d array as required by cdist func.
        xa = x[algorithm].iloc[0].reshape(1,-1)
        
        # row stack the other hashes
        xb = x.iloc[1:][algorithm].values
        xb = np.row_stack(xb)

        return dist.cdist(xa, xb, metric=metric) 

    def fit(self, data):
        self.files_ = data['filename'].unique()
        self.n_files_ = len(self.files_)

        distances = []

        for a in self.le_a.classes_:
            for m in self.le_m.classes_:

                # Compute the distances for each filename
                grp_dists = data.groupby(['filename']).apply(
                    func=self.intradistance,
                    algorithm=a,
                    metric=m)
                
                # Stack each distance into rows
                grp_dists = np.row_stack(grp_dists)
                
                # Get the integer labels for algo and metric
                a_label = self.le_a.transform(a.ravel())[0]
                m_label = self.le_m.transform(m.ravel())
                
                grp_dists = np.column_stack([
                    self.files_, # fileA
                    self.files_, # fileB (same in intra!)
                    np.repeat(a_label,self.n_files_),
                    np.repeat(m_label, self.n_files_),
                    np.repeat(self.set_class, self.n_files_),
                    grp_dists])
                distances.append(grp_dists)

        distances = np.concatenate(distances)

        # Create the dataframe output
        cols = ['fileA','fileB','algo','metric','class', *self.le_t.classes_[:-1]]
        distances = pd.DataFrame(distances, columns=cols)
        distances['orig'] = 0
        
        # set int columns accordingly
        int_cols = cols[:5]
        distances[int_cols] = distances[int_cols].astype(int)
        
        # Convert distances to similarities
        sim_cols = distances.columns[5:]
        distances[sim_cols] = 1-distances[sim_cols]
    
        return distances

class InterDistance():
    def __init__(self, le_t, le_m, le_a, set_class=1, n_samples=100, random_state=42):
        self.le_t = le_t
        self.le_m = le_m
        self.le_a = le_a
        self.set_class = set_class
        self.n_samples = n_samples
        self.random_state = random_state

    def interdistance(self, x, algorithm, metric):
        # get hashes into a 2d array
        hashes = np.row_stack(x[algorithm])
        
        # return pairwise distances of all combinations
        return dist.pdist(hashes, metric)

    def fit(self, data):
        # Get the label used to encode 'orig'
        orig_label = self.le_t.transform(np.array(['orig']).ravel())[0]
        
        # Assert sufficient data to sample from.
        assert len(data[data['transformation'] == orig_label]) >= self.n_samples

        # Pick the samples
        self.samples_ = data[data['transformation'] == orig_label].sample(
             self.n_samples, 
             random_state=self.random_state)['filename'].values
        
        # Subset the data
        subset = data[data['filename'].isin(self.samples_)]

        # Create unique pairs matching the output of scipy.spatial.distances.pdist
        self.pairs_ = np.array([c for c in combinations(subset['filename'].unique(), 2)])
        
        # Count the number of unique pairs
        self.n_pairs_ = len(self.pairs_)
        
        # List to hold distances while looping over algorithms and metrics
        distances = []
    
        # Do the math using Pandas groupby
        for a in self.le_a.classes_:
            for m in self.le_m.classes_:
                # Compute distances for each group of transformations
                grp_dists = subset.groupby(['transformation']).apply(
                    self.interdistance, #type:ignore
                    algorithm=a,
                    metric=m)
                
                # Transpose to create rows of observations
                X_dists = np.transpose(np.row_stack(grp_dists.values))
                
                # Get the integer labels for algo and metric
                a_label = self.le_a.transform(a.ravel())[0]
                m_label = self.le_m.transform(m.ravel())

                # Add columns with pairs of the compared observations
                X_dists = np.column_stack([
                    self.pairs_,
                    np.repeat(a_label,self.n_pairs_),
                    np.repeat(m_label,self.n_pairs_),
                    np.repeat(self.set_class, self.n_pairs_),
                    X_dists])
                
                # Add the results to the distances array
                distances.append(X_dists)

        # Flatten the distances array
        distances = np.concatenate(distances)
        
        # Create the dataframe output
        cols = ['fileA','fileB','algo','metric','class', *self.le_t.classes_]
        distances = pd.DataFrame(distances, columns=cols)
        
        # Set datatype to int on all non-distance columns
        int_cols = cols[:5]
        distances[int_cols] = distances[int_cols].astype(int)
    
        # Convert distances to similarities
        sim_cols = distances.columns[5:]
        distances[sim_cols] = 1-distances[sim_cols]
        
        return distances