from PyQt5.QtCore import pyqtSignal, pyqtSlot


class SignalSlotCachable:
    def __init__(self):
        self._signal_slot_maps = {}

    def connect_signal_slot(self, signal: pyqtSignal, slot: pyqtSlot):
        if slot is None or signal is None:
            return False

        connection = signal.connect(slot)
        if signal in self._signal_slot_maps.keys():
            _slots = self._signal_slot_maps[signal]
            _slots.append(connection)
        else:
            self._signal_slot_maps[signal] = [connection]

        return True

    def disconnect_signal_slot(self, signal) -> bool:
        if signal not in self._signal_slot_maps.keys():
            return False
        for connection in self._signal_slot_maps[signal]:
            signal.disconnect(connection)
            connection = None
        self._signal_slot_maps[signal] = [p for p in self._signal_slot_maps[signal] if p is not None]
        return True

    def disconnect_all(self):
        for item in self._signal_slot_maps.items():
            for connection in item[1]:
                item[0].disconnect(connection)
        self._signal_slot_maps = {}
