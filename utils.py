from timeit import default_timer as timer
from collections import deque

class PrettyDeque(deque):
    def __repr__(self):
        s = ', '.join('%.2f'%x for x in self)
        return '[%s]'%s

class StopWatch(object):
    def __init__(self):
        self.stored = 0
        self.start_time = -1

    def pause(self):
        self.stored = timer() - self.start_time
        self.start_time = -1

    def start(self):
        self.start_time = timer()

    def reset(self):
        self.stored = 0
        self.start_time = -1

    def time(self):
        return self.stored + (0 if self.start_time == -1 else timer() - self.start_time)
