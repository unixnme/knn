import multiprocessing as mp
import argparse

import numpy as np


class Processor:
    def __init__(self, input_file:str, batch_size:int):
        self.filename = input_file
        self.batch_size = batch_size

    def run(self):
        raise NotImplementedError

    def preprocess(self, line:str):
        tokens = line.split('\t')
        return tokens[0], tuple(float(x) for x in tokens[1].split('|'))

    def batch_process(self, ids, vectors):
        return np.asarray(ids), np.asarray(vectors)


class SingleProcessor(Processor):
    def run(self):
        result = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            batch = []
            for idx, line in enumerate(f):
                batch.append(self.preprocess(line))

                if len(batch) >= self.batch_size:
                    result.append(tuple(zip(*batch)))
                    batch = []

        if len(batch) > 0:
            result.append(tuple(zip(*batch)))

        return [self.batch_process(*batch) for batch in result]


class PoolProcessor(Processor):
    def __init__(self, input_file:str, batch_size:int, nproc:int):
        super().__init__(input_file, batch_size)
        self.nproc = nproc

    def run(self):
        result = []
        with mp.Pool(self.nproc) as pool:
            with open(self.filename, 'r', encoding='utf-8') as f:
                lines = []
                for idx, line in enumerate(f):
                    lines.append(line)

                    if len(lines) >= self.batch_size:
                        result.append(tuple(zip(*pool.map(self.preprocess, lines))))
                        lines = []

            if len(lines) > 0:
                result.append(tuple(zip(*pool.map(self.preprocess, lines))))

        return [self.batch_process(*batch) for batch in result]


class FileReader(mp.Process):
    def __init__(self, filename:str, task_queue:mp.Queue, nproc:int):
        self.filename = filename
        self.task_queue = task_queue
        self.nproc = nproc
        super().__init__()

    def run(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                self.task_queue.put(line)
        for _ in range(self.nproc):
            self.task_queue.put(None)


class LineProcessor(mp.Process):
    def __init__(self, task_queue:mp.Queue, result_queue:mp.Queue):
        self.task_queue = task_queue
        self.result_queue = result_queue
        super().__init__()

    def run(self):
        for line in iter(self.task_queue.get, None):
            tokens = line.split('\t')
            vector = tuple(float(x) for x in tokens[1].split('|'))
            self.result_queue.put((tokens[0], vector))
        self.result_queue.put(None)

class AdvancedProcessor(PoolProcessor):
    def run(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        FileReader(self.filename, self.task_queue, self.nproc).start()
        for _ in range(self.nproc):
            LineProcessor(self.task_queue, self.result_queue).start()

        batch = []
        result = []
        for _ in range(self.nproc):
            for x in iter(self.result_queue.get, None):
                batch.append(x)
                if len(batch) >= self.batch_size:
                    result.append(tuple(zip(*batch)))
                    batch = []
        if len(batch) > 0:
            result.append(tuple(zip(*batch)))

        return [self.batch_process(*batch) for batch in result]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', type=int, required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--advanced', action='store_true')
    args = parser.parse_args()

    if args.nproc <= 0:
        result = SingleProcessor(args.input_file, args.batch_size).run()
    elif not args.advanced:
        result = PoolProcessor(args.input_file, args.batch_size, args.nproc).run()
    else:
        result = AdvancedProcessor(args.input_file, args.batch_size, args.nproc).run()
    print(result)