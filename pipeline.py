import multiprocessing as mp
from typing import Callable
import argparse
import logging


def line_process(line):
    tokens = line.split('\t')
    vector = tuple(float(x) for x in tokens[1].split('|'))
    return tokens[0], vector


def batch_process(batch):
    return [line_process(line) for line in batch]


class FileReader(mp.Process):
    def __init__(self, filename:str, queue:mp.SimpleQueue):
        super().__init__()
        self.filename = filename
        self.queue = queue
        logging.debug(f'initiating {self}')

    def run(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                self.queue.put(line)
            self.queue.put(None)
        logging.debug(f'exiting {self}')


class Batcher(mp.Process):
    def __init__(self, queue_in: mp.SimpleQueue, queue_out: mp.SimpleQueue, batch_size:int):
        super().__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.batch_size = batch_size
        logging.debug(f'initiating {self}')

    def run(self):
        batch = []
        for item in iter(self.queue_in.get, None):
            batch.append(item)
            if len(batch) >= self.batch_size:
                self.queue_out.put(batch)
                batch = []
        if batch:
            self.queue_out.put(batch)
        self.queue_out.put(None)
        logging.debug(f'exiting {self}')


class Pipeline(mp.Process):
    def __init__(self, queue_in:mp.SimpleQueue, queue_out:mp.SimpleQueue, work:Callable, counter):
        super().__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.work = work
        self.counter = counter
        logging.debug(f'initiating {self}')

    def run(self):
        for item in iter(self.queue_in.get, None):
            self.queue_out.put(self.work(item))
        logging.debug(f"{self}: reading from qin complete")
        self.counter.value -= 1
        count = self.counter.value
        logging.debug(f'remaining workers: {count}')
        if count == 0:
            # i'm the last worker; will shut down
            self.queue_out.put(None)
        else:
            # friends still working, one of whom will shut down
            self.queue_in.put(None)
        logging.debug(f'exiting {self}')


class BackgroundPool:
    def __init__(self, num_workers:int, work:Callable, queue_in:mp.SimpleQueue, queue_out:mp.SimpleQueue):
        self.num_workers = num_workers
        self.work = work
        self.queue_in = queue_in
        self.queue_out = queue_out
        # lock not needed, since queue_in is already synchronized
        self.counter = mp.Value('i', lock=False)

    def start(self):
        self.counter.value = self.num_workers
        for _ in range(self.num_workers):
            Pipeline(self.queue_in, self.queue_out, self.work, self.counter).start()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument('--nproc', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()

    q1 = mp.SimpleQueue()
    FileReader(args.file, q1).start()

    q2 = mp.SimpleQueue()
    BackgroundPool(args.nproc, line_process, q1, q2).start()

    q3 = mp.SimpleQueue()
    Batcher(q2, q3, args.batch_size).start()
    
    count = 0
    for item in iter(q3.get, None):
        count += 1
        logging.debug(count)
