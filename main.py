import argparse
import sys
import numpy as np


class NearestNeighborBatch:
    def __init__(self, db_ids, db, k:int):
        '''
        :param db_ids: unique id for each db vector
        :param db: [db_size, dim] vector array
        '''
        assert len(db_ids) == len(db)
        self.db_ids = db_ids
        self.db = db
        self.db_size = len(db)
        self.dim = len(db[0])
        self.k = k

    def find(self, qid, query):
        '''

        :param qid: unique query id for each query vector
        :param query: [q_size, dm] vector array
        '''
        raise NotImplementedError

    def format(self, D, I, qids):
        qids = np.asarray(qids)
        db_ids = np.asarray(self.db_ids)[I]
        return np.stack([qids.repeat(self.k), db_ids.reshape(-1), D.reshape(-1)], axis=1)

def merge_shards(shards, k:int):
    '''
    :param shards: array of shard, where each shard has the form
        [qid, urlhash, inner_product] repeated by k
    :return: [qid, urlhash] repeated by k
    '''
    result = []
    shards = np.stack(shards, axis=-1)
    for start in range(0, len(shards), k):
        qids = shards[start:start+k, 0, 0]
        urlhash = shards[start:start+k, 1]
        values = shards[start:start+k, -1]
        idx = np.argsort(shards[start:start+k, -1].astype(np.float32).reshape(-1))[::-1]
        result.append((qids, urlhash.reshape(-1)[idx[:k]], values.reshape(-1)[idx[:k]]))
    return np.asarray(result).transpose((0,2,1)).reshape(-1, 3)



class FaissNearestNeighborBatch(NearestNeighborBatch):
    def __init__(self, db_ids, db, k:int):
        assert isinstance(db, np.ndarray)
        super().__init__(db_ids, db, k)

    def find(self, qids, query):
        index = faiss.IndexFlatIP(self.dim)
        index.add(self.db)
        D, I = index.search(query, self.k)
        return self.format(D, I, qids)


class PytorchNearestNeighborBatch(NearestNeighborBatch):
    def __init__(self, db_ids, db, k:int):
        assert isinstance(db, torch.FloatTensor)
        super().__init__(db_ids, db, k)
        self.db = self.db.t()

    def find(self, qids, query):
        '''
        :param qids: list of ids
        :param query: torch tensor of query vectors
        '''
        D, I = torch.topk(query @ self.db, self.k, dim=-1)
        return self.format(D.numpy(), I.numpy(), qids)


def change_type(ids, vector:np.ndarray, backend:str):
    if backend == 'faiss':
        return ids, vector
    elif backend == 'pytorch':
        return ids, torch.from_numpy(vector)


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


def write_out(result):
    '''
    :param result: array of (qid, urlhash)
    '''
    for line in result:
        print('\t'.join(line))


if __name__ == '__main__':
    default_config = {
        'batch_size': 1024,
        'sep': '|',
        'topk': 5,
        'max_mem': 16,
        'shard_size': 1000,
    }

    parser = argparse.ArgumentParser('brute force nearest neighbor search')
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'],
                        help=f'query batch size; default: {default_config["batch_size"]}')
    parser.add_argument('--db_file', required=True)
    parser.add_argument('--shard_size', type=int, default=default_config['shard_size'])
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

    shards = []
    if args.backend == "faiss":
        Solver = FaissNearestNeighborBatch
    elif args.backend == "pytorch":
        Solver = PytorchNearestNeighborBatch
    else:
        raise ValueError

    print(f"estimate mem: {estimate_memory(args.batch_size, args.shard_size, args.dim):.2f}GB", file=sys.stderr)

    for shard_idx, db in enumerate(read_batch_from_file(args.db_file, args.shard_size, args.sep)):
        print(f"processing shard {shard_idx}...", file=sys.stderr)
        db = change_type(*db, args.backend)
        solver = Solver(*db, args.topk)
        q_batch_result = []
        for qbatch_idx, q in enumerate(read_batch_from_file(args.query_file, args.batch_size, args.sep)):
            print(f"processing query batch {qbatch_idx}...", file=sys.stderr)
            q = change_type(*q, args.backend)
            q_batch_result.append(solver.find(*q))
        shards.append(np.concatenate(q_batch_result))
    result = merge_shards(shards, args.topk)
    write_out(result)