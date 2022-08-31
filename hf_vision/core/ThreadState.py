from enum import Enum


class ThreadStateType(Enum):
    none = 0
    init = none + 1
    ready = init + 1
    running = ready + 1
    pause = running + 1
    stop = pause + 1
    finalized = stop + 1


class ThreadState(object):
    def __init__(self):
        super().__init__()
        self.state = ThreadStateType.none
        self.user_info = None
