import os
import glob
import json
import urllib.request
from pathlib import Path

import numpy as np
import cv2
import yt_dlp
import gradio as gr
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure

from config import Config as CONFIG, DetectConfig


plt.style.use('dark_background')
plt.rcParams.update({'figure.figsize': (12, 20)})
plt.rcParams.update({'font.size': 9})


def download_model(model_name: str, models_dir: Path, model_urls: dict[str, str]) -> str:
    model_path = models_dir / model_name
    if not model_path.exists():
        urllib.request.urlretrieve(model_urls[model_name], model_path)
    return str(model_path)


def detect_image(detect_config: DetectConfig) -> np.ndarray:
    detect_func = detect_config.model.predict
    detect_kwargs = dict(
        source=detect_config.source,
        conf=detect_config.conf,
        iou=detect_config.iou,
        save=detect_config.save_image_predicts,
        verbose=detect_config.verbose,
        project=detect_config.results_dir,
    )
    if detect_config.detect_mode == 'Tracking':
        detect_func = detect_config.model.track
        tracker = CONFIG.TRACKERS[detect_config.tracker_name]
        detect_kwargs['tracker'] = tracker
        detect_kwargs['persist'] = True
    detections = detect_func(**detect_kwargs)
    np_image = detections[0].plot()
    return np_image


def detect_webcam(
        np_image: np.ndarray,
        model_state: dict[str, YOLO],
        conf: float,
        iou: float,
        detect_mode: str,
        tracker_name: str,
    ) -> np.ndarray:
    detect_config = DetectConfig(
        source=np_image,
        model=model_state['model'],
        conf=conf,
        iou=iou,
        detect_mode=detect_mode,
        tracker_name=tracker_name,
    )
    new_np_image = detect_image(detect_config)
    return new_np_image
 

def detect_video(detect_config: DetectConfig) -> tuple[UltralyticsResults, int, int]:
    video_path = detect_config.source
    if 'youtube.com' in video_path or 'youtu.be' in video_path:
        gr.Progress()(0.001, desc='Downloading video from YouTube...')
        ydl_opts = {'format': 'bestvideo[height<=720]'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info_dict = ydl.extract_info(video_path, download=True)
            video_path = ydl.prepare_filename(video_info_dict)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    detect_func = detect_config.model.predict
    detect_kwargs = dict(
        source=video_path,
        conf=detect_config.conf,
        iou=detect_config.iou,
        save_txt=True,
        save_conf=True,
        stream=True,
        save=detect_config.save_video_predicts,
        verbose=detect_config.verbose,
        project=detect_config.results_dir,
    )
    if detect_config.detect_mode == 'Tracking':
        detect_func = detect_config.model.track
        tracker = CONFIG.TRACKERS[detect_config.tracker_name]
        detect_kwargs['tracker'] = tracker

    generator = detect_func(**detect_kwargs)
    frames_count = 0
    for result in generator:
        frames_count += 1
        yield result, frames_count, total_frames


def get_csv_annotate(
        result_video_path: np.ndarray | Path | None,
        curr_csv_annotations_path: str | None,
        double_call_flag: bool = False,
    ) -> Path | None:

    if not isinstance(result_video_path, Path) or double_call_flag:
        return curr_csv_annotations_path

    is_tracking = 'track' in result_video_path.parent.stem
    txts_path = result_video_path.parent / 'labels'
    escaped_pattern = glob.escape(result_video_path.stem)
    matching_txts_path = sorted(txts_path.glob(f'{escaped_pattern}_*.txt'), key=os.path.getmtime)
    if not matching_txts_path:
        gr.Info('No txt detection results')
        return curr_csv_annotations_path
    
    df_list = []
    for txt_path in matching_txts_path:
        frame_number = int(txt_path.stem.rsplit('_')[-1])
        with open(txt_path) as file:
            df_rows = file.readlines()
            for df_row in df_rows:
                df_row = map(float, df_row.split())
                row_list = [frame_number, *df_row]
                if is_tracking and len(row_list) == 7:
                    row_list.append(None)
                df_list.append(row_list)

    column_names = ['frame_number', 'class_label', 'x', 'y', 'w', 'h', 'conf']
    if is_tracking:
        column_names.append('track_id')

    df = pd.DataFrame(df_list, columns=column_names)
    df.class_label = df.class_label.astype(int)
    if 'track_id' in df.columns:
        df.track_id = df.track_id.astype('Int64')

    class_name_series = df.class_label.map(CONFIG.YOLO_CLASS_NAMES)
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
    df['box_detected'] = df['class_name'].notna().astype(int)

    result_csv_path = f'{result_video_path.parent / result_video_path.stem}_annotations.csv'
    df.to_csv(result_csv_path, index=False)
    return Path(result_csv_path)


def get_matplotlib_fig(csv_annotations_path: str) -> MatplotlibFigure:
    df = pd.read_csv(csv_annotations_path)
    df_clean = df.dropna(subset=['class_name'])

    fig, axes = plt.subplots(7, 1, figsize=(10, 20), constrained_layout=True)

    sns.histplot(data=df_clean['conf'], kde=True, ax=axes[0])
    axes[0].set_title('Распределение уверенности детекций')
    axes[0].set_xlabel('Уверенность')
    axes[0].set_ylabel('Количество обнаружений')

    sns.boxplot(data=df_clean, x='class_name', y='conf', ax=axes[1])
    axes[1].set_title('Распределение уверенности детекции по классам')
    axes[1].set_xlabel('Класс объекта')
    axes[1].set_ylabel('Уверенность')
    # axes[1].tick_params(axis='x', labelrotation=45)

    sns.countplot(
        data=df_clean,
        x='class_name',
        hue='class_name',
        order=df_clean['class_name'].value_counts().index,
        palette='viridis',
        legend=False,
        ax=axes[2],
        )
    axes[2].set_title('Количество обнаружений объектов по классам')
    axes[2].set_xlabel('Класс объекта')
    axes[2].set_ylabel('Количество')

    face_count_per_frame = df.groupby('frame_number')['box_detected'].sum()
    axes[3].plot(face_count_per_frame.index, face_count_per_frame.values, marker='o', linestyle='-')
    axes[3].set_title('Частота обнаружения объектов по кадрам')
    axes[3].set_xlabel('Номер кадра')
    axes[3].set_ylabel('Количество обнаруженных объектов')

    face_count_per_frame = df.groupby('frame_sec')['box_detected'].sum()
    axes[4].plot(face_count_per_frame.index, face_count_per_frame.values, marker='o', linestyle='-')
    axes[4].set_title('Частота обнаружения объектов по секундам')
    axes[4].set_xlabel('Время (сек)')
    axes[4].set_ylabel('Количество обнаруженных объектов')

    sns.scatterplot(
        data=df_clean,
        x='frame_sec',
        y='class_name',
        hue='class_name',
        palette='deep',
        s=50,
        alpha=0.6,
        legend=True,
        ax=axes[5],
        )
    axes[5].set_title('Временная шкала обнаружения объектов по классам')
    axes[5].set_xlabel('Время видео (секунды)')
    axes[5].set_ylabel('Эмоция')
    axes[5].grid(True, linestyle='--', alpha=0.7)
    axes[5].legend(title='Классы объектов', bbox_to_anchor=(1.05, 1), loc='upper left')

    emotion_timeline = df.pivot_table(index='frame_sec', columns='class_name', aggfunc='size', fill_value=0)
    emotion_timeline.plot(kind='area', stacked=True, ax=axes[6])
    axes[6].set_title('Динамика обнаружения классов во времени')
    axes[6].set_xlabel('Время видео (секунды)')
    axes[6].set_ylabel('Количество детекций')
    axes[6].grid(True, linestyle='--', alpha=0.7)
    axes[6].legend(title='Классы объектов', bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig