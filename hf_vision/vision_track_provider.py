import time
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QPixmap
from torch.backends import cudnn

from hf_vision.qt import utils
from hf_vision.regional_tracking import RegionalDetectTrackerM2, BoundaryLine, Area, checkLineCrosses, resetLineCrosses, \
    checkAreaIntrusion, TrackingObject, drawBoundaryLines, drawAreas
from strong_sort import StrongSORT
from strong_sort.utils.parser import get_config
from track import boundaryLines
from vision_tracker_config import VisionTrackingConfig
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.datasets import VID_FORMATS
from yolov5.utils.general import check_suffix, check_img_size, check_file, set_logging, non_max_suppression, \
    scale_coords, xyxy2xywh
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device


class VisionTrackingResult(object):
    def __init__(self, objects: [TrackingObject]):
        super().__init__()
        self.objects = objects


class VisionTrackingProvider(QThread):
    _cv_pred_update_signal = pyqtSignal(VisionTrackingResult)  # signal for output vision-tracking pred data
    _tracking_img_signal = pyqtSignal(QPixmap)

    def connect_cv_pred_signal(self, slot):
        self._cv_pred_update_signal.connect(slot) \
            if slot is not None \
            else None

    def connect_tracking_vision_signal(self, slot):
        self._tracking_img_signal.connect(slot) \
            if slot is not None \
            else None

    def disconnect_signal(self):
        self._cv_pred_update_signal.disconnect()
        self._tracking_img_signal.disconnect()

    def __init__(self, config: VisionTrackingConfig, boundaryLines: [BoundaryLine], areas: [Area]):
        super().__init__()
        self._is_running_flag = False
        self.cv_raw_img = None
        self.config = config
        self.boundaryLines = boundaryLines
        self.areas = areas
        self.half = False

    def run(self):
        self._setup_vision_track_engine()
        set_logging()

        assert self.device is not None

        self._is_running_flag = True

        while self._is_running_flag:
            if self.cv_raw_img is None:
                time.sleep(0)
                continue

            with torch.no_grad():
                im = letterbox(self.cv_raw_img, new_shape=self.config.imgsz, stride=self.stride, auto=True)[0]
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)

                im = torch.from_numpy(im).to(self.device)

                im = im.half() if self.half else im.float()
                im /= 255.0  # convert 0~255 to 0.0~1.0
                if len(im.shape) == 3:
                    im = im[None]

                # Inference
                pred = self.model(im, augment=False, visualize=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.config.conf_threshold, iou_thres=0.45,
                                           classes=self.config.track_classes, agnostic=False, max_det=1000)

                # Process detections
                for i, det in enumerate(pred):
                    self.curr_frames[i] = self.cv_raw_img.copy()
                    origin_img = self.curr_frames[i].copy()
                    origin_height, origin_width = origin_img.shape[:2]

                    annotator = Annotator(origin_img, line_width=2, pil=not ascii)

                    if self.cfg.STRONGSORT.ECC:
                        self.reid_sort_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

                    if det is not None and len(det):
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], origin_img.shape).round()

                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]

                        self.reid_outputs[i] = self.reid_sort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), origin_img.copy())
                        if len(self.reid_outputs[i]):
                            combine_outputs = zip(self.reid_outputs[i], confs)
                            for j, (output, conf) in enumerate(combine_outputs):
                                bboxes = output[0:4]
                                id = int(output[4])
                                cls = int(output[5])

                                self.regional_detector.try_tracking(pos=[bboxes[0], bboxes[1], bboxes[2], bboxes[3]],
                                                                    id=id,
                                                                    feature=None,
                                                                    conf=conf)

                                label = f'{id}'
                                if not self.config.hide_classes:
                                    label += f' {self.names[cls]}'
                                if not self.config.hide_conf:
                                    label += f' {conf:.2f}'

                                if label is not None and len(label):
                                    annotator.box_label(bboxes, label, color=colors(cls, True))
                    else:
                        self.reid_sort_list[i].increment_ages()

                    self.regional_detector.trackObjects([])
                    self.regional_detector.evictTimeoutObjectFromDB()
                    self.regional_detector.drawTrajectory(origin_img, self.regional_detector.objects)

                    checkLineCrosses(boundaryLines, self.regional_detector.objects)
                    drawBoundaryLines(origin_img, boundaryLines)
                    resetLineCrosses(boundaryLines)

                    checkAreaIntrusion(self.areas, self.regional_detector.objects)
                    drawAreas(origin_img, self.areas)

                    # print(f'{len(self.regional_detector.objects)}')
                    self._cv_pred_update_signal.emit(VisionTrackingResult(self.regional_detector.objects)) \
                        if len(self.regional_detector.objects) \
                        else None

                    # emit the result with the tracked labels.
                    annotated_img = annotator.result()
                    self._tracking_img_signal.emit(utils.convert_cv_qt(cv_img=annotated_img, width=origin_width, height=origin_height)) \
                        if self.config.using_tracker_vision \
                        else None

                    self.prev_frames[i] = self.curr_frames[i]
            self.cv_raw_img = None

        print('vision tracker provider stops....???')

    def stop(self):
        self._is_running_flag = False

    # receive image in cv format from outside
    @pyqtSlot(np.ndarray)
    def on_cv_frame_update_slot(self, cv_img):
        self.cv_raw_img = cv_img

    def _setup_vision_track_engine(self):
        self.source = str(self.config.source)
        is_file = Path(self.source).suffix[1:] in [VID_FORMATS]
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)

        self.device = select_device(self.config.device)
        self.half &= self.device.type != 'cpu'
        track_weight = str(self.config.yolo_weights[0]) \
            if isinstance(self.config.yolo_weights, list) \
            else self.config.yolo_weights
        classify, suffix, suffixes = False, Path(track_weight).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(track_weight, suffixes)  # check weights have acceptable suffix
        self.pt, self.onnx, self.tflite, self.pb, self.saved_model = (suffix == x for x in suffixes)  # backend booleans

        self.model = torch.jit.load(track_weight) \
            if 'torchscript' in track_weight \
            else attempt_load(self.config.yolo_weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names \
            if hasattr(self.model, 'module') \
            else self.model.names

        self.model.half() if self.half else None  # to FP16
        self.imgzs = check_img_size(self.config.imgsz, s=self.stride)

        if webcam:
            cudnn.benchmark = True
            # CODING HERE FOR WEB CAMERA
            assert False
        else:
            nr_sources = 1

        self.vid_path, = [None] * nr_sources

        self.cfg = get_config()
        self.cfg.merge_from_file(self.config.reid_sort_config)

        # Create as many strong sort instances as there are video sources
        self.reid_sort_list = []
        for i in range(nr_sources):
            self.reid_sort_list.append(
                StrongSORT(
                    self.config.reid_sort_weights,
                    self.device,
                    max_dist=self.cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=self.cfg.STRONGSORT.MAX_AGE,
                    n_init=self.cfg.STRONGSORT.N_INIT,
                    nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,
                )
            )
        self.reid_outputs = [None] * nr_sources

        # regional-detect
        self.regional_detector = RegionalDetectTrackerM2(conf_thres=self.config.conf_threshold)

        if self.pt and self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, *self.imgzs).to(self.device).type_as(next(self.model.parameters())))  # run once

        self.curr_frames, self.prev_frames = [None] * nr_sources, [None] * nr_sources
