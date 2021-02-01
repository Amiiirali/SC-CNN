import numpy as np
import networkx as nx
from scipy.spatial import distance
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors
from networkx.algorithms.clique import find_cliques

def find_local_maxima_candidate(proximity, dist=8, thresh=0.3, radius=2):

    # Row-vise
    prox_flat_row = proximity.flatten(order='C')
    row_peaks, _  = find_peaks(prox_flat_row, height=thresh, distance=dist)
    row_idx       = np.unravel_index(row_peaks, proximity.shape, order='C')
    row_idx       = np.array([[row_idx[0][num], row_idx[1][num]] for num in range(len(row_idx[0]))])

    # Col-vise
    prox_flat_col = proximity.flatten(order='F')
    col_peaks, _  = find_peaks(prox_flat_col, height=thresh, distance=dist)
    col_idx       = np.unravel_index(col_peaks, proximity.shape, order='F')
    col_idx       = np.array([[col_idx[0][num], col_idx[1][num]] for num in range(len(col_idx[0]))])

    neigh = NearestNeighbors(n_neighbors=1, radius=radius)

    neigh.fit(row_idx)
    idx         = neigh.kneighbors(col_idx, return_distance=False)
    row_col_idx = row_idx[idx.reshape(idx.shape[0],)]

    neigh.fit(col_idx)
    idx         = neigh.kneighbors(row_idx, return_distance=False)
    col_row_idx = col_idx[idx.reshape(idx.shape[0],)]

    total  = np.concatenate((row_col_idx, col_row_idx), axis=0)
    _, idx = np.unique(total, axis=0, return_index=True)
    total  = total[idx]

    return total


def remove_clique(proximity, candidates, dist):

    graph_dist = distance.cdist(candidates, candidates, 'euclidean')

    graph_dist[graph_dist==0]    = dist + 1
    graph_dist[graph_dist<dist]  = 1
    graph_dist[graph_dist>=dist] = 0

    G           = nx.from_numpy_array(graph_dist)
    max_cliques = list(find_cliques(G))

    new_candidates = np.empty(shape=[len(max_cliques), 2])

    for idx, clique in enumerate(max_cliques):

        points = candidates[clique].astype(int)
        values = proximity[points[:,0], points[:,1]]

        new_candidates[idx, :] = points[values.argmax()]

    return new_candidates


def postprocess(proximity, dist=8, thresh=0.3, radius=2):

    candidates = find_local_maxima_candidate(proximity, dist=dist, thresh=thresh,
                                             radius=radius)
    num_cand = len(candidates)

    # round(0.5) is 1 in Matlab, but 0 in Python
    for d in range(round(dist/2+0.00001), dist+1):

        if num_cand > 1:
            candidates = remove_clique(proximity, candidates, d)

        if len(candidates) < num_cand:
            num_cand = len(candidates)
        else:
            break

    # candidates[:,[0,1]] = candidates[:,[1,0]]
    return candidates
