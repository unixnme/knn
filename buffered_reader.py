import argparse
import multiprocessing as mp
import time
import numpy as np


class FileReader(mp.Process):
    def __init__(self, filename:str, queue:mp.SimpleQueue):
        super().__init__()
        self.filename = filename
        self.queue = queue

    def run(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            sum = 0.0
            count = 0
            for line in f:
                self.queue.put(line)

        self.queue.put(None)


class LineProcessor(mp.Process):
    def __init__(self, iq:mp.SimpleQueue, oq:mp.SimpleQueue):
        super().__init__()
        self.iq = iq
        self.oq = oq

    def run(self):
        for line in iter(self.iq.get, None):
            vector = [float(x) for x in line.split('\t')[1].split('|')]
            self.oq.put(np.linalg.norm(vector).item())
        self.oq.put(None)


class Adder(mp.Process):
    def __init__(self, iq: mp.SimpleQueue, oq: mp.SimpleQueue):
        super().__init__()
        self.iq = iq
        self.oq = oq

    def run(self):
        sum = 0.0
        for s in iter(self.iq.get, None):
            sum += s
        self.oq.put(sum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('time', type=float)
    args = parser.parse_args()

    q1 = mp.SimpleQueue()
    q2 = mp.SimpleQueue()
    q3 = mp.SimpleQueue()
    FileReader('db.tsv', q1).start()
    LineProcessor(q1, q2).start()
    Adder(q2, q3).start()

    time.sleep(args.time)
    print(q3.get())

