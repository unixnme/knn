import argparse
import numpy as np


class NearestNeighborBatch:
    def __init__(self, db_ids, db):
        '''
        :param db_ids: unique id for each db vector
        :param db: [db_size, dim] vector array
        '''
        assert len(db_ids) == len(db)
        self.db_ids = db_ids
        self.db = db
        self.db_size = len(db)
        self.dim = len(db[0])

    def topk(self, qid, query, k:int):
        '''

        :param qid: unique query id for each query vector
        :param query: [q_size, dm] vector array
        :param k: top k ids
        '''
        raise NotImplementedError


class FaissNearestNeighborBatch(NearestNeighborBatch):
    def __init__(self, db_ids, db):
        assert isinstance(db, np.ndarray)
        super().__init__(db_ids, db)

    def topk(self, qid, query, k:int):
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
            v = np.asarray([float(x) for x in cols[1].split(sep)], dtype=np.float32)
            ids.append(id)
            vectors.append(v)

            if len(ids) >= batch_size:
                yield ids, np.stack(vectors, axis=0)
                ids, vectors = [], []
    if len(ids) > 0:
        yield ids, np.stack(vectors, axis=0)


def estimate_memory(n_q:int, n_db:int, dim:int):
    return ((n_q + n_db) * dim + n_q * n_db) * 4 / 1e9


if __name__ == '__main__':
    default_config = {
        'batch_size': 1024,
        'sep': '|',
        'topk': 5,
        'max_mem': 16,
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
    parser.add_argument('--max_mem', type=int, default=default_config['max_mem'],
                        help=f'maximum memory in GB; default: {default_config["max_mem"]}')

    args = parser.parse_args()
    if args.backend == 'faiss':
        import faiss
    else:
        import torch

    for db in read_batch_from_file(args.db_file, 1000, args.sep):
        if args.backend == 'faiss':
            solver = FaissNearestNeighborBatch(*db)
        else:
            raise NotImplementedError
        for q in read_batch_from_file(args.query_file, args.batch_size, args.sep):
            idx = solver.topk(*q, args.topk)