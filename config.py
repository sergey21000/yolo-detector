import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


class Config:
    MODELS_DIR: Path = Path('models')
    MODELS_DIR.mkdir(exist_ok=True)

    YOLO_CLASS_NAMES: dict[str, str] = json.loads(Path('yolo_classes.json').read_text())
    YOLO_CLASS_NAMES: dict[int, str] = {int(k): v for k, v in YOLO_CLASS_NAMES.items()}

    MODEL_URLS: dict[str, str] = {
        'yolov11n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
        'yolov11s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
        'yolov11m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
        'yolov11l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt',
        'yolov11x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
    }

    MODEL_NAMES: list[str] = list(MODEL_URLS.keys())
    IMAGE_EXTENSIONS: list[str] = ['.jpg', '.jpeg', '.png']
    VIDEO_EXTENSIONS: list[str] = ['.mp4', '.avi']
    DETECT_MODE_NAMES: list[str] = ['Detection', 'Tracking']
    TRACKERS: dict[str, str] = {'ByteTrack': 'bytetrack.yaml', 'BoT-SORT': 'botsort.yaml'}
    TRACKER_NAMES: list[str] = list(TRACKERS.keys())
    WEBCAM_TIME_LIMIT: int = 30


@dataclass
class DetectConfig:
    source: str | np.ndarray
    model: YOLO
    conf: float
    iou: float
    detect_mode: str
    tracker_name: str
