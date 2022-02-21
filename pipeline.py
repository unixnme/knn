import sys
import multiprocessing as mp
import argparse


class Pipeline(mp.Process):
    def __init__(self, queue_in:mp.Queue, queue_out:mp.Queue, work):
        super().__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.work = work

    def run(self):
        for item in iter(self.queue_in.get, None):
            # print(f"{self}: {self.queue_out.qsize()}", file=sys.stderr)
            self.queue_out.put(self.work(item))
        print(f"{self}: complete; #output queue size: {self.queue_out.qsize()}", file=sys.stderr)
        self.queue_out.put(None)


class Adapter(mp.Process):
    def __init__(self, queue_in:mp.Queue, queue_out:mp.Queue, nproc_in:int, nproc_out:int):
        super().__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.nproc_in = nproc_in
        self.nproc_out = nproc_out

    def run(self):
        for _ in range(self.nproc_in):
            for item in iter(self.queue_in.get, None):
                self.queue_out.put(item)
        print(f"{self}: complete; #output queue size: {self.queue_out.qsize()}", file=sys.stderr)
        for _ in range(self.nproc_out):
            self.queue_out.put(None)


class Batcher(mp.Process):
    def __init__(self, queue_in: mp.Queue, queue_out: mp.Queue, batch_size:int):
        super().__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.batch_size = batch_size

    def run(self):
        batch = []
        for item in iter(self.queue_in.get, None):
            batch.append(item)
            if len(batch) >= self.batch_size:
                self.queue_out.put(batch)
                batch = []
        if batch:
            self.queue_out.put(batch)
        print(f"{self}: complete; #output queue size: {self.queue_out.qsize()}", file=sys.stderr)
        self.queue_out.put(None)


class MultiWorkerPipeline:
    def __init__(self, queue_in:mp.Queue, queue_out: mp.Queue, work, nproc:int = None):
        self.queue_in = queue_in
        self.qin = mp.Queue() # intermediate
        self.queue_out = queue_out
        self.qout = mp.Queue() # intermediate
        self.nproc = nproc or mp.cpu_count()
        self.work = work

    def start(self):
        Adapter(self.queue_in, self.qin, 1, self.nproc).start()
        Adapter(self.qout, self.queue_out, self.nproc, 1).start()

        for _ in range(self.nproc):
            Pipeline(self.qin, self.qout, self.work).start()


class FileReader(mp.Process):
    def __init__(self, filename:str, queue:mp.Queue):
        super().__init__()
        self.filename = filename
        self.queue = queue

    def run(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                self.queue.put(line)
            print(f"{self}: complete; #output queue size: {self.queue.qsize()}", file=sys.stderr)
            self.queue.put(None)


def line_process(line):
    tokens = line.split('\t')
    vector = tuple(float(x) for x in tokens[1].split('|'))
    return tokens[0], vector


def batch_process(batch):
    return [line_process(line) for line in batch]


def f(q):
    for item in iter(q.get, None):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument('--nproc', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()

    q1 = mp.Queue()
    FileReader(args.file, q1).start()
    q2 = mp.Queue()
    MultiWorkerPipeline(q1, q2, line_process, args.nproc).start()
    q3 = mp.Queue()
    Batcher(q2, q3, args.batch_size).start()

    mp.Process(target=f, args=(q3,)).start()