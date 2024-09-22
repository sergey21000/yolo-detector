import os
import glob
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

import torch
import pandas as pd
import numpy as np
import cv2
import yt_dlp
import gradio as gr
from ultralytics import YOLO


YOLO_CLASS_NAMES = json.loads(Path('yolo_classes.json').read_text())


def download_model(model_name: str, models_dir: Path, models: dict) -> str:
    model_path = models_dir / model_name
    if not model_path.exists():
        urllib.request.urlretrieve(models[model_name], model_path)
    return str(model_path)


def detect_image(image_path: str, model: YOLO, conf: float, iou: float) -> np.ndarray:
    gr.Progress()(0.5, desc='Детекция изображения...')
    detections = model.predict(source=image_path, conf=conf, iou=iou)
    np_image = detections[0].plot()
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    return np_image


def detect_video(video_path_or_url: str, model: YOLO, conf: float, iou: float) -> Tuple[Path, Path]:
    progress = gr.Progress()
    video_path = video_path_or_url
    if 'youtube.com' in video_path_or_url or 'youtu.be' in video_path_or_url:
        progress(0.001, desc='Загрузка видео с YouTube...')
        ydl_opts = {'format': 'bestvideo[height<=720]'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info_dict = ydl.extract_info(video_path_or_url, download=True)
            video_path = ydl.prepare_filename(video_info_dict)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    generator = model.predict(
        source=video_path,
        conf=0.5,
        iou=0.5,
        save=True,
        save_txt=True,
        save_conf=True,
        stream=True,
        verbose=False,
        )

    frames_count = 0
    for result in generator:
        frames_count += 1
        progress((frames_count, num_frames), desc=f'Детекция видео, шаг {frames_count}/{num_frames}')

    file_name = Path(result.path).with_suffix('.avi').name
    result_video_path = Path(result.save_dir) / file_name
    Path(video_path).unlink(missing_ok=True)
    return result_video_path


def get_csv_annotate(result_video_path: Path) -> str:
    if not isinstance(result_video_path, Path):
        return None

    txts_path = result_video_path.parent / 'labels'
    escaped_pattern = glob.escape(result_video_path.stem)
    matching_txts_path = sorted(txts_path.glob(f'{escaped_pattern}_*.txt'), key=os.path.getmtime)

    df_list = []
    for txt_path in matching_txts_path:
        frame_number = int(txt_path.stem.rsplit('_')[-1])
        with open(txt_path) as file:
            df_rows = file.readlines()
            for df_row in df_rows:
                df_row = map(float, df_row.split())
                df_list.append((frame_number, *df_row))

    column_names = ['frame_number', 'class_label', 'x', 'y', 'w', 'h', 'conf']
    df = pd.DataFrame(df_list, columns=column_names)

    df.class_label = df.class_label.astype(int)
    class_name_series = df.class_label.map(YOLO_CLASS_NAMES)
    df.insert(loc=1, column='class_name', value=class_name_series)

    cap = cv2.VideoCapture(str(result_video_path))
    frames_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    frame_sec_series = df.frame_number / frames_fps
    df.insert(loc=1, column='frame_sec', value=frame_sec_series)

    full_frames = pd.DataFrame({'frame_number': range(total_frames)})
    df = pd.merge(full_frames, df, on='frame_number', how='outer')
    df.frame_sec = df.frame_number / frames_fps

    result_csv_path = f'{result_video_path.parent / result_video_path.stem}_annotations.csv'
    df.to_csv(result_csv_path, index=False)
    return result_csv_path