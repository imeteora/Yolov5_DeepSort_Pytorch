import time
from typing import List

import cv2
import numpy as np
from munkres import Munkres
from scipy.spatial import distance

from hf_vision.regional_tracking import RawObject


class RegionalDetectTracker:
    def __init__(self):
        self.object_id = 0
        self.timeout = 3  # sec
        self.clear_db()
        self.similarityThreshold = 0.6
        pass

    def clear_db(self):
        self.object_db = []

    def evictTimeoutObjectFromDB(self):
        # discard time out objects
        now = time.monotonic()
        for object in self.object_db:
            if object.time + self.timeout < now:
                self.object_db.remove(object)  # discard feature vector from DB
                print("Discarded  : id {}".format(object.id))

    # objects = list of object class
    def trackObjects(self, objects: List[RawObject]):
        # if no object found, skip the rest of processing
        if len(objects) == 0:
            return

        now = time.monotonic()

        # If any object is registered in the db, assign registered ID to the most similar object in the current image
        if len(self.object_db) > 0:
            # Create a matrix of cosine distance
            cos_sim_matrix = [
                [distance.cosine(objects[j].feature, self.object_db[i].feature) for j in range(len(objects))]
                for i in range(len(self.object_db))
            ]
            # solve feature matching problem by Hungarian assignment algorithm
            combination = Munkres().compute(cos_sim_matrix)

            # assign ID to the object pairs based on assignment matrix
            for dbIdx, objIdx in combination:
                if distance.cosine(objects[objIdx].feature, self.object_db[dbIdx].feature) < self.similarityThreshold:
                    # assign an ID
                    objects[objIdx].id = self.object_db[dbIdx].id
                    # update the feature vector in DB with the latest vector (to make tracking easier)
                    self.object_db[dbIdx].feature = objects[objIdx].feature
                    # update last found time
                    self.object_db[dbIdx].time = now
                    # record position history as trajectory
                    self.object_db[dbIdx].trajectory.append(objects[objIdx].anchor_pt)
                    objects[objIdx].trajectory = self.object_db[dbIdx].trajectory

        # Register the new objects which has no ID yet
        for obj in list(filter(lambda e: e.id == -1, objects)):
            # no similar objects is registered in feature_db
            obj.id = self.object_id
            obj.time = now
            # position history for trajectory line
            obj.trajectory = [obj.anchor_pt]
            # register a new feature to the db
            self.object_db.append(obj)
            self.object_id += 1

    def drawTrajectory(self, img, objects):
        filtered = list(filter(lambda obj: len(obj.trajectory) > 2, objects))
        for obj in filtered:
            cv2.polylines(img, np.array([obj.trajectory], np.int32), False, (0, 0, 0), 4)
