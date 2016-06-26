import numpy as np
import tables


random_state = np.random.RandomState(1999)

def make_random_cluster_points(n_samples, random_state=random_state):
    mu_options = np.array([(-1, -1), (1, 1), (1, -1), (-1, 1)])
    sigma = 0.2
    mu_choices = random_state.randint(0, len(mu_options), size=n_samples)
    means = mu_options[mu_choices]
    return means + np.random.randn(n_samples, 2) * sigma, mu_choices


sample_data, sample_clusters = make_random_cluster_points(10000)
hdf5_path = "my_data.hdf5"
hdf5_file = tables.openFile(hdf5_path, mode='w')
data_storage = hdf5_file.createArray(hdf5_file.root, 'data', sample_data)
clusters_storage = hdf5_file.createArray(hdf5_file.root, 'clusters', sample_clusters)
hdf5_file.close()
