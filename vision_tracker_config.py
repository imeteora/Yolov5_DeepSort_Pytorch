from enum import Enum
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


class TrackingError(Enum):
    ok = 0
    no_yolo_weights = ok + 1
    no_strong_sort_weights = no_yolo_weights + 1
    no_strong_sort_config = no_strong_sort_weights + 1
    no_source = no_strong_sort_config + 1


class VisionTrackingConfig(object):
    def __init__(self, device: str = 'cuda:0', source: str = None, yolo_weights: str = None, reid_sort_weights: str = None,
                 reid_sort_config: str = ROOT / 'strong_sort/configs/strong_sort.yaml', image_size=(640, 640),
                 half=False, enable_dnn=False, track_classes=[0], using_tracker_vision=True, hide_classes=True,
                 hide_conf=False, conf_threshold=0.5):
        super().__init__()
        self.device = device if device is not None and len(device) else 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.source = source  # video file, web-cam url, local cam: 0
        self.yolo_weights = yolo_weights
        self.reid_sort_weights = reid_sort_weights
        self.reid_sort_config = reid_sort_config
        self.imgsz = image_size if image_size else (640, 640)
        self.half = half if half is not None else False
        self.dnn = enable_dnn if enable_dnn is not None else False
        self.track_classes = track_classes
        self.using_tracker_vision = using_tracker_vision
        self.hide_classes = hide_classes if hide_classes is not None else False
        self.hide_conf = hide_conf if hide_conf is not None else False
        self.conf_threshold = conf_threshold if conf_threshold is not None else 0.5

    @property
    def has_yolo_weights(self) -> bool:
        return self.yolo_weights is not None and len(self.yolo_weights)

    @property
    def has_reid_sort_weights(self) -> bool:
        return self.reid_sort_weights is not None and len(self.reid_sort_weights)

    @property
    def has_reid_sort_config(self) -> bool:
        return self.reid_sort_config is not None and len(self.reid_sort_config)

    @property
    def status(self) -> TrackingError:
        if not self.has_yolo_weights:
            return TrackingError.no_yolo_weights
        elif not self.has_reid_sort_weights:
            return TrackingError.no_strong_sort_weights
        elif not self.has_reid_sort_config:
            return TrackingError.no_strong_sort_config
        else:
            return TrackingError.ok

    @property
    def is_ready(self) -> bool:
        return self.status is TrackingError.ok
