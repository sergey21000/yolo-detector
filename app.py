import shutil
from pathlib import Path
from typing import List, Dict, Union, Tuple, Literal, Optional

import numpy as np
import gradio as gr
from gradio.components.base import Component
from ultralytics import YOLO

from utils import download_model, detect_image, detect_video, get_csv_annotate


# ======================= MODEL ===================================

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    'yolov11n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
    'yolov11s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
    'yolov11m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
    'yolov11l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt',
    'yolov11x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
}
MODEL_NAMES = list(MODELS.keys())

model_path = download_model(MODEL_NAMES[0], MODELS_DIR, MODELS)
default_model = YOLO(model_path)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
VIDEO_EXTENSIONS = ['.mp4', '.avi']


# =================== ADDITIONAL INTERFACE FUNCTIONS ========================

def change_model(model_state: Dict[str, YOLO], model_name: str):
    progress = gr.Progress()
    progress(0.3, desc='Downloading the model')
    model_path = download_model(model_name)
    progress(0.7, desc='Model initialization')
    model_state['model'] = YOLO(model_path)
    return f'Model {model_name} initialized'


def detect(file_path: str, file_link: str, model_state: Dict[str, YOLO], conf: float, iou: float):
    model = model_state['model']
    if file_link:
        file_path = file_link

    file_ext = f'.{file_path.rsplit(".")[-1]}'
    if file_ext in IMAGE_EXTENSIONS:
        np_image = detect_image(file_path, model, conf, iou)
        return np_image, "Detection complete, opening image..."
    elif file_ext in VIDEO_EXTENSIONS or 'youtube.com' in file_link:
        video_path = detect_video(file_path, model, conf, iou)
        return video_path, "Detection complete, converting and opening video..."
    else:
        gr.Info('Invalid image or video format...')
        return None, None

# =================== INTERFACE COMPONENTS ============================

def get_output_media_components(detect_result: Optional[Union[np.ndarray, str, Path]] = None):
    visible = isinstance(detect_result, np.ndarray)
    image_output = gr.Image(
        value=detect_result if visible else None,
        type="numpy",
        width=640,
        height=480,
        visible=visible,
        label='Output',
        )
    visible = isinstance(detect_result, (str, Path))
    video_output = gr.Video(
        value=detect_result if visible else None,
        width=640,
        height=480,
        visible=visible,
        label='Output',
        )
    clear_btn = gr.Button(
        value='Clear',
        scale=0,
        visible=detect_result is not None,
        )
    return image_output, video_output, clear_btn


def get_download_csv_btn(csv_annotations_path: Optional[Path] = None):
    download_csv_btn = gr.DownloadButton(
        label='Download csv annotations for video',
        value=csv_annotations_path,
        scale=0,
        visible=csv_annotations_path is not None,
        )
    return download_csv_btn

# =================== APPINTERFACE ==========================

css = '''
.gradio-container { width: 70% !important }
'''
with gr.Blocks(css=css) as demo:
    gr.HTML("""<h3 style='text-align: center'>YOLOv11 Detector</h3>""")
    
    model_state = gr.State({'model': default_model})
    detect_result = gr.State(None)
    csv_annotations_path = gr.State(None)

    with gr.Row():
        with gr.Column():
            file_path = gr.File(file_types=['image', 'video'], file_count='single', label='Select an image or video')
            file_link = gr.Textbox(label='Direct link to image or YouTube link')
            model_name = gr.Radio(choices=MODEL_NAMES, value=MODEL_NAMES[0], label='Select YOLO model')
            conf = gr.Slider(0, 1, value=0.5, step=0.05, label='Confidence')
            iou = gr.Slider(0, 1, value=0.7, step=0.1, label='IOU')
            status_message = gr.Textbox(value='Ready to go', label='Status')
            detect_btn = gr.Button('Detect', interactive=True)

        with gr.Column():
            image_output, video_output, clear_btn = get_output_media_components()
            download_csv_btn = get_download_csv_btn()

    model_name.change(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=[detect_btn],
    ).then(
        fn=change_model,
        inputs=[model_state, model_name],
        outputs=[status_message],
    ).success(
        fn=lambda: gr.update(interactive=True),
        inputs=None,
        outputs=[detect_btn],
    )

    detect_btn.click(
        fn=detect,
        inputs=[file_path, file_link, model_state, conf, iou],
        outputs=[detect_result, status_message],
    ).success(
        fn=get_output_media_components,
        inputs=[detect_result],
        outputs=[image_output, video_output, clear_btn],
    ).then(
        fn=lambda: 'Ready to go',
        inputs=None,
        outputs=[status_message],
    ).then(
        fn=get_csv_annotate,
        inputs=[detect_result],
        outputs=[csv_annotations_path],
    ).success(
        fn=get_download_csv_btn,
        inputs=[csv_annotations_path],
        outputs=[download_csv_btn],
    )

    def clear_results_dir(detect_result):
        if isinstance(detect_result, Path):
            shutil.rmtree(detect_result.parent, ignore_errors=True)

    clear_components = [image_output, video_output, clear_btn, download_csv_btn]
    clear_btn.click(
        fn=lambda: [gr.update(visible=False) for _ in range(len(clear_components))],
        inputs=None,
        outputs=clear_components,
    ).then(
        fn=clear_results_dir,
        inputs=[detect_result],
        outputs=None,
    ).then(
        fn=lambda: (None, None),
        inputs=None,
        outputs=[detect_result, csv_annotations_path]
        )

    gr.HTML("""<h3 style='text-align: center'>
    <a href="https://github.com/sergey21000/yolo-detector" target='_blank'>GitHub Page</a></h3>
    """)
    
demo.launch(server_name='0.0.0.0')  # debug=True