import time

from regional_tracking import RawObject, RegionalDetectTracker


class RegionalDetectTrackerM2(RegionalDetectTracker):

    def __init__(self):
        super().__init__()
        self.timeout = 3  # sec
        self.clear_db()

    def clear_db(self):
        self.object_db = {}

    def evictTimeoutObjectFromDB(self):
        now = time.monotonic()
        self.object_db = {key: val for key, val in self.object_db.items() if val.time + self.timeout >= now}

    @property
    def objects(self) -> [RawObject]:
        return [obj for (obj_id, obj) in self.object_db.items()]

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

    def trackObjects(self, objects: [RawObject]):
        # if len(objects) <= 0:
        #     return
        #
        # now = time.monotonic()
        #
        # for obj in objects:
        #     if obj.id in self.object_db:
        #         self.object_db[obj.id].time = now
        #         self.object_db[obj.id].trajectory.append(obj.anchor_pt)
        #         obj.trajectory = self.object_db[obj.id].trajectory
        #     else:
        #         obj.time = now
        #         obj.trajectory = [obj.anchor_pt]
        #         self.object_db[obj.id] = obj
        pass

    def drawTrajectory(self, img, objects):
        super().drawTrajectory(img=img, objects=objects)
