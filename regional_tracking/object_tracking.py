import time

import cv2
import numpy as np
from munkres import Munkres
from scipy.spatial import distance


# ------------------------------------
# Object tracking

class region_object:
    def __init__(self, pos, feature, id=-1):
        self.feature = feature
        self.id = id
        self.trajectory = []
        self.time = time.monotonic()
        self.pos = pos


class objectTracker:
    def __init__(self):
        self.objectid = 0
        self.timeout = 3  # sec
        self.clearDB()
        self.similarityThreshold = 0.4
        pass

    def clearDB(self):
        self.objectDB = []

    def evictTimeoutObjectFromDB(self):
        # discard time out objects
        now = time.monotonic()
        for object in self.objectDB:
            if object.time + self.timeout < now:
                self.objectDB.remove(object)  # discard feature vector from DB
                print("Discarded  : id {}".format(object.id))

    # objects = list of object class
    def trackObjects(self, objects):
        # if no object found, skip the rest of processing
        if len(objects) == 0:
            return

        # If any object is registred in the db, assign registerd ID to the most similar object in the current image
        if len(self.objectDB) > 0:
            # Create a matix of cosine distance
            cos_sim_matrix = [[distance.cosine(objects[j].feature, self.objectDB[i].feature)
                               for j in range(len(objects))] for i in range(len(self.objectDB))]
            # solve feature matching problem by Hungarian assignment algorithm
            hangarian = Munkres()
            combination = hangarian.compute(cos_sim_matrix)

            # assign ID to the object pairs based on assignment matrix
            for dbIdx, objIdx in combination:
                if distance.cosine(objects[objIdx].feature, self.objectDB[dbIdx].feature) < self.similarityThreshold:
                    objects[objIdx].id = self.objectDB[dbIdx].id  # assign an ID
                    self.objectDB[dbIdx].feature = objects[
                        objIdx].feature  # update the feature vector in DB with the latest vector (to make tracking easier)
                    self.objectDB[dbIdx].time = time.monotonic()  # update last found time
                    xmin, ymin, xmax, ymax = objects[objIdx].pos
                    self.objectDB[dbIdx].trajectory.append(
                        [(xmin + xmax) // 2, (ymin + ymax) // 2])  # record position history as trajectory
                    objects[objIdx].trajectory = self.objectDB[dbIdx].trajectory

        # Register the new objects which has no ID yet
        for obj in objects:
            if obj.id == -1:  # no similar objects is registred in feature_db
                obj.id = self.objectid
                self.objectDB.append(obj)  # register a new feature to the db
                self.objectDB[-1].time = time.monotonic()
                xmin, ymin, xmax, ymax = obj.pos
                self.objectDB[-1].trajectory = [
                    [(xmin + xmax) // 2, (ymin + ymax) // 2]]  # position history for trajectory line
                obj.trajectory = self.objectDB[-1].trajectory
                self.objectid += 1

    def drawTrajectory(self, img, objects):
        for obj in objects:
            if len(obj.trajectory) > 1:
                cv2.polylines(img, np.array([obj.trajectory], np.int32), False, (0, 0, 0), 4)
