import glob
import faiss
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Iterable, Tuple
from numpy import ndarray
import numpy as np
from .__main__ import pickle_load, write_ranking

def change_ids(indices):
    n, m = indices.shape
    indices_new = [[0 for i in range(m)] for j in range(n)]
    for i in range(n):
        for j in range(m):
            if indices[i][j].startswith('table_'):
                idx = int(indices[i][j][len('table_'):]) + 30000000
            else:
                idx = int(indices[i][j])
            global_dict[idx] = indices[i][j]
            indices_new[i][j] = idx
    return np.array(indices_new)
def combine_faiss_results(results: Iterable[Tuple[ndarray, ndarray]]):
    rh = None
    for scores, indices in results:
        if rh is None:
            print(f'Initializing Heap. Assuming {scores.shape[0]} queries.')
            rh = faiss.ResultHeap(scores.shape[0], scores.shape[1])
        #print(indices.shape)
        indices = change_ids(indices)
        #print(indices.shape)
        rh.add_result(-scores, indices)
    rh.finalize()
    corpus_scores, corpus_indices = -rh.D, rh.I

    return corpus_scores, corpus_indices


def main():
    parser = ArgumentParser()
    parser.add_argument('--score_dir', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--save_ranking_to', required=True)
    
    args = parser.parse_args()

    partitions = glob.glob(f'{args.score_dir}/*')

    corpus_scores, corpus_indices = combine_faiss_results(map(pickle_load, tqdm(partitions)))

    _, q_lookup = pickle_load(args.query)
    write_ranking(corpus_indices, corpus_scores, q_lookup, args.save_ranking_to, global_dict=global_dict)


if __name__ == '__main__':
    global_dict = {}
    main()
