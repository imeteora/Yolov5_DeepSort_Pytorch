import time
from typing import List

from hf_vision.regional_tracking.raw_object import RawObject
from hf_vision.regional_tracking.regional_detect_tracker import RegionalDetectTracker
from hf_vision.regional_tracking.tracking_object import TrackingObject


class RegionalDetectTrackerM2(RegionalDetectTracker):

    @property
    def objects(self) -> [RawObject]:
        return [obj for (obj_id, obj) in self.object_db.items()]

    def __init__(self, conf_thres: float = 0.5):
        super().__init__()
        self.object_db = None
        self.timeout = 3  # sec
        self.conf_thres = conf_thres
        self.clear_db()

    def clear_db(self):
        self.object_db = {}

    def evictTimeoutObjectFromDB(self):
        now = time.monotonic()
        self.object_db = {key: val for key, val in self.object_db.items() if val.time + self.timeout >= now}

    def try_tracking(self, pos: List[int], id: int, feature=None, conf: float = 0.5):
        if conf < self.conf_thres:
            return

        if found := self.object_with(obj_id=id):
            found.update(pos=pos)
        else:
            new_obj = TrackingObject(pos, feature=feature, id=id)
            self.append_object(obj=new_obj)

    def object_with(self, obj_id: int) -> any:
        return self.object_db[obj_id] \
            if obj_id in self.object_db \
            else None

    def append_object(self, obj):
        if obj.id in self.object_db:
            return

        obj.time = time.monotonic()
        obj.trajectory = [obj.anchor_pt]
        self.object_db[obj.id] = obj

    def trackObjects(self, objects: List[RawObject]):
        pass

    def drawTrajectory(self, img, objects):
        super().drawTrajectory(img=img, objects=objects)
