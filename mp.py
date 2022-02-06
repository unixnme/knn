import multiprocessing as mp
import argparse
import numpy as np


class Processor:
    def __init__(self, input_file:str, batch_size:int):
        self.filename = input_file
        self.batch_size = batch_size

    def run(self):
        raise NotImplementedError

    def process(self, line:str):
        tokens = line.split('\t')
        return tokens[0], (float(x) for x in tokens[1].split('|'))


class SingleProcessor(Processor):
    def run(self):
        result = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            ids, batch = [], []
            for idx, line in enumerate(f):
                id, vector = self.process(line)
                ids.append(id)
                batch.append(vector)

                if len(ids) >= self.batch_size:
                    result.append((ids, np.asarray(batch)))
                    ids, batch = [], []

        if len(ids) > 0:
            result.append((ids, np.asarray(batch)))

        return result


def process(line:str):
    tokens = line.split('\t')
    id = tokens[0]
    vector = [float(x) for x in tokens[1].split('|')]
    return id, vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', type=int, required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()

    if args.nproc <= 0:
        result = SingleProcessor(args.input_file, args.batch_size).run()
    else:
        pass