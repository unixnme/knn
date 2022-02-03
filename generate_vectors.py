import argparse
import numpy as np
from uuid import uuid4


ids = set()
def generate_id(uuid:bool) -> str:
    if uuid:
        while True:
            id = uuid4().hex
            if id not in ids: break
    else:
        id = str(len(ids))
    ids.add(id)
    return id


def generate_vector(dim:int, normalize:bool) -> np.ndarray:
    x = np.random.randn(dim)
    if normalize:
        x = x / np.linalg.norm(x)
    return x


def vec2str(vector:np.ndarray, sep:str, prec:int) -> str:
    s = '{:.' + str(prec) + 'f}'
    return sep.join([s.format(x) for x in vector])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate vectors')
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--uuid', action='store_true')
    parser.add_argument('-n', type=int, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--sep', default='|')
    parser.add_argument('--precision', default=3)

    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)
    for i in range(args.n):
        id = generate_id(args.uuid)
        v = generate_vector(args.dim, args.normalize)
        print(f'{id}\t{vec2str(v, args.sep, args.precision)}')
