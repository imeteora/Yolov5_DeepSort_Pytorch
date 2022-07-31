import time

import cv2
import numpy as np
from munkres import Munkres
from scipy.spatial import distance


# ------------------------------------
# Object tracking

class RegionObject:
    def __init__(self, pos, feature, id=-1):
        self.feature = feature
        self.id = id
        self.trajectory = []
        self.time = time.monotonic()
        self.pos = pos  # (left, top, right, bottom)

    def anchor_pt(self) -> [int]:
        p0 = (self.pos[0] + self.pos[2]) // 2
        p1 = self.pos[3]  # (self.pos[1] + self.pos[3]) // 2
        return [p0, p1]

    @property
    def left(self) -> int:
        return int(self.pos[0])

    @property
    def top(self) -> int:
        return int(self.pos[1])

    @property
    def right(self) -> int:
        return int(self.pos[2])

    @property
    def bottom(self) -> int:
        return int(self.pos[3])

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class RegionDetectTracker:
    def __init__(self):
        self.object_id = 0
        self.timeout = 3  # sec
        self.clear_db()
        self.similarityThreshold = 0.6
        pass

    def clear_db(self):
        self.objectDB = []

    def evictTimeoutObjectFromDB(self):
        # discard time out objects
        now = time.monotonic()
        for object in self.objectDB:
            if object.time + self.timeout < now:
                self.objectDB.remove(object)  # discard feature vector from DB
                print("Discarded  : id {}".format(object.id))

    # objects = list of object class
    def trackObjects(self, objects: [RegionObject]):
        # if no object found, skip the rest of processing
        if len(objects) == 0:
            return

        tm_current = time.monotonic()

        gen_anchor = lambda xmin, ymin, xmax, ymax: [(xmin + xmax) // 2, int(ymax)]

        # If any object is registered in the db, assign registered ID to the most similar object in the current image
        if len(self.objectDB) > 0:
            # Create a matrix of cosine distance
            cos_sim_matrix = [
                [distance.cosine(objects[j].feature, self.objectDB[i].feature) for j in range(len(objects))]
                for i in range(len(self.objectDB))
            ]
            # solve feature matching problem by Hungarian assignment algorithm
            combination = Munkres().compute(cos_sim_matrix)

            # assign ID to the object pairs based on assignment matrix
            for dbIdx, objIdx in combination:
                if distance.cosine(objects[objIdx].feature, self.objectDB[dbIdx].feature) < self.similarityThreshold:
                    # assign an ID
                    objects[objIdx].id = self.objectDB[dbIdx].id
                    # update the feature vector in DB with the latest vector (to make tracking easier)
                    self.objectDB[dbIdx].feature = objects[objIdx].feature
                    # update last found time
                    self.objectDB[dbIdx].time = tm_current
                    # record position history as trajectory
                    xmin, ymin, xmax, ymax = objects[objIdx].pos
                    self.objectDB[dbIdx].trajectory.append(gen_anchor(xmin, ymin, xmax, ymax))
                    objects[objIdx].trajectory = self.objectDB[dbIdx].trajectory

        # Register the new objects which has no ID yet
        for obj in list(filter(lambda e: e.id == -1, objects)):
            # no similar objects is registered in feature_db
            obj.id = self.object_id
            obj.time = tm_current
            xmin, ymin, xmax, ymax = obj.pos
            # position history for trajectory line
            obj.trajectory = [gen_anchor(xmin, ymin, xmax, ymax)]
            # register a new feature to the db
            self.objectDB.append(obj)
            self.object_id += 1

    def drawTrajectory(self, img, objects):
        filtered = list(filter(lambda obj: len(obj.trajectory) > 2, objects))
        for obj in filtered:
            cv2.polylines(img, np.array([obj.trajectory], np.int32), False, (0, 0, 0), 4)
