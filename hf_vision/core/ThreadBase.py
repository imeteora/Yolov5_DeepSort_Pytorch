import time

from PyQt5.QtCore import QThread


class ThreadBase(QThread):

    def __init__(self):
        super().__init__()
        self._fps_frame_time = 0.0
        self._is_running = False
        self.fps = 60.0

    @property
    def fps(self) -> float:
        return self._fps

    @fps.setter
    def fps(self, new_value):
        self._fps = min(120.0, max(0, new_value))
        self._fps_frame_time = 1000.0 / self._fps

    @classmethod
    def time_interval(cls) -> int:
        return int(round(time.time() * 1000))

    def run(self) -> None:
        self.pre_main()

        _latest_frame_time = self.time_interval()
        while self._is_running:
            _current_frame_time = self.time_interval()
            if _current_frame_time < _latest_frame_time + self._fps_frame_time:
                time.sleep(0)
                continue
            _latest_frame_time = _current_frame_time

            if not self.main():
                break

        self.post_main()

    def start(self, priority: QThread.Priority = ...):
        super().start(QThread.NormalPriority)
        self._is_running = True

    def pre_main(self):
        print(f'pls implement {self.pre_main.__name__} in sub-classes.')
        pass

    def main(self):
        print(f'pls implement {self.main.__name__} in sub-classes.')
        pass

    def post_main(self):
        print(f'pls implement {self.post_main.__name__} in sub-classes.')
        pass

    def stop(self):
        self._is_running = False
        self.exit(0)
