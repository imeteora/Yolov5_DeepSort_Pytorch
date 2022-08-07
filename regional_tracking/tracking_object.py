import time

import cv2
import numpy as np

from regional_tracking import RawObject


class TrackingObject(RawObject):
    def __init__(self, pos, feature, id: int = -1):
        super().__init__(pos=pos, feature=feature, id=id)
        self.crossed_lines = []

        self.current_measurement = self.last_measurement = np.array((2, 1), np.float32)
        self.current_prediction = self.last_prediction = np.array((2, 1), np.float32)

        self.kalman = cv2.KalmanFilter(4, 2)    # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                np.float32) # 状态转移矩阵
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32)*0.0004 # 系统过
        self.pt0 = super().anchor_pt

    def update(self, pos: [int]):
        self.pos = pos
        self.time = time.monotonic()

        pt = super().anchor_pt
        x, y = pt[0] - self.pt0[0], pt[1] - self.pt0[1]

        self.last_prediction = self.current_prediction  # 把当前预测存储为上一次预测
        self.last_measurement = self.current_measurement  # 把当前测量存储为上一次测量
        self.current_measurement = np.array([[np.float32(x)], [np.float32(y)]])  # 当前测量
        self.kalman.correct(self.current_measurement)  # 用当前测量来校正卡尔曼滤波器
        self.current_prediction = self.kalman.predict()  # 计算卡尔曼预测值，作为当前预测

        # lmx, lmy = self.last_measurement[0], self.last_measurement[1]  # 上一次测量坐标
        # cmx, cmy = self.current_measurement[0], self.current_measurement[1]  # 当前测量坐标
        # lpx, lpy = self.last_prediction[0], self.last_prediction[1]  # 上一次预测坐标
        # cpx, cpy = self.current_prediction[0], self.current_prediction[1]  # 当前预测坐标

        # 加入新的轨迹点（经过kalman计算之后）
        if len(self.trajectory) > 30:
            self.trajectory = self.trajectory[1:]
        self.trajectory.append(self.anchor_pt)

    @property
    def anchor_pt(self) -> [int]:
        p0, p1 = self.current_prediction[0], self.current_prediction[1]
        return [int(p0 + self.pt0[0]), int(p1 + self.pt0[1])]
