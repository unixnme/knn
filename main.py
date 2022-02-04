import argparse
import numpy as np


class NearestNeighborBatch:
    def __init__(self, db):
        '''
        :param db: [db_size, dim] array
        '''
        self.db = db
        self.db_size = len(db)
        self.dim = len(db[0])

    def topk(self, query, k:int):
        raise NotImplementedError


class FaissNearestNeighborBatch(NearestNeighborBatch):
    def __init__(self, db):
        assert isinstance(db, np.ndarray)
        super().__init__(db)

    def topk(self, query, k:int):
        index = faiss.IndexFlatL2(self.dim)
        index.add(self.db)
        D, I = index.search(query, k)
        return I


def read_batch_from_file(filename:str, batch_size:int, sep:str):
    with open(filename, 'r', encoding='utf-8') as f:
        ids, vectors = [], []
        for line in f:
            cols = line.split('\t')
            id = cols[0]
            v = np.asarray([float(x) for x in cols[1].split(sep)])
            ids.append(id)
            vectors.append(v)

            if len(ids) >= batch_size:
                yield ids, np.stack(vectors, axis=0)
                ids, vectors = [], []
    if len(ids) > 0:
        yield ids, np.stack(vectors, axis=0)


if __name__ == '__main__':
    default_config = {
        'batch_size': 1024,
        'sep': '|',
        'topk': 5,
    }

    parser = argparse.ArgumentParser('brute force nearest neighbor search')
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'],
                        help=f'query batch size; default: {default_config["batch_size"]}')
    parser.add_argument('--db_file', required=True)
    parser.add_argument('--sep', default=default_config['sep'],
                        help=f'separator for vector representation; default: {default_config["sep"]}')
    parser.add_argument('--dim', type=int, required=True, help='expected vector dimension; all non-compliant entry will be ignored')
    parser.add_argument('--backend', choices=['pytorch', 'faiss'], default='pytorch',
                        help='brute force search backend')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    parser.add_argument('--topk', type=int, default=default_config['topk'],
                        help=f'top k nearest vectors per query; default: {default_config["topk"]}')

    args = parser.parse_args()
    if args.backend == 'faiss':
        import faiss
    else:
        import torch

    for db in read_batch_from_file(args.db_file, 5000, args.sep):
        print(len(db[0]))