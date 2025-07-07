import shutil
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from fastrtc import WebRTC
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults
from matplotlib.figure import Figure as MatplotlibFigure

from dotenv import load_dotenv
load_dotenv(dotenv_path='gradio_env')

from config import Config as CONFIG, DetectConfig
from utils import (
    download_model,
    detect_image, 
    detect_video, 
    get_csv_annotate, 
    get_matplotlib_fig,
)


# ======================= MODEL ===================================


model_path = download_model(CONFIG.MODEL_NAMES[0], CONFIG.MODELS_DIR, CONFIG.MODEL_URLS)
default_model = YOLO(model_path)


# =================== ADDITIONAL INTERFACE FUNCTIONS ========================

def change_model(model_state: dict[str, YOLO], model_name: str) -> str:
    progress = gr.Progress()
    progress(0.3, desc='Downloading the model')
    model_path = download_model(model_name, CONFIG.MODELS_DIR, CONFIG.MODEL_URLS)
    progress(0.7, desc='Model initialization')
    model_state['model'] = YOLO(model_path)
    return f'Model {model_name} initialized'


def reinit_model(model_state: dict[str, YOLO], model_name: str, reinit: bool) -> None:
    if reinit:
        change_model(model_state, model_name)


def detect(
        file_path: str,
        file_link: str,
        model_state: dict[str, YOLO],
        conf: float,
        iou: float,
        detect_mode: str,
        tracker_name: str,
    ) -> tuple[np.ndarray | UltralyticsResults | None, str | dict]:
    
    if not file_path and not file_link:
        yield None, 'Empty input'
        return
    if file_link:
        file_path = file_link
    detect_config = DetectConfig(
        source=file_path,
        model=model_state['model'],
        conf=conf,
        iou=iou,
        detect_mode=detect_mode,
        tracker_name=tracker_name,
    )
    file_ext = f'.{file_path.rsplit(".")[-1]}'
    if file_ext in CONFIG.IMAGE_EXTENSIONS:
        gr.Progress()(0.5, desc='Image detection...')
        np_image = detect_image(detect_config)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        yield np_image, gr.skip()
        return
    elif file_ext in CONFIG.VIDEO_EXTENSIONS or 'youtube.com' in file_path:
        generator = detect_video(detect_config)
        progress = gr.Progress()
        update_every_frame = 20
        for result, frames_count, total_frames in generator:
            progress(
                progress=(frames_count, total_frames),
                desc=f'Video detection, step {frames_count}/{total_frames}',
                total=total_frames,
            )
            if frames_count == 1:
                yield result, gr.skip()
            if frames_count % update_every_frame == 0 or frames_count == total_frames:
                yield result, gr.skip()
    else:
        yield None, 'Invalid image or video format...'
        return


def get_video_path_from_ultralytics_result(result: np.ndarray | Path | None) -> np.ndarray | Path | None:
    if isinstance(result, UltralyticsResults):
        file_name = Path(result.path).with_suffix('.avi').name
        result_video_path = Path(result.save_dir) / file_name
        return result_video_path
    return result


def get_ready_status() -> str:
    return 'Ready to Go'


def update_detect_status(detect_result, status_message) -> str:
    if detect_result is None:
        status = status_message
        gr.Info(status)
    elif isinstance(detect_result, np.ndarray):
        status = 'Detection complete, opening image...'
    elif isinstance(detect_result, Path):
        status = 'Detection complete, converting and opening video...'
    else:
        status = get_ready_status()
    return status


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


def clear_results_dir(detect_result: np.ndarray | Path | None) -> None:
    if isinstance(detect_result, Path):
        shutil.rmtree(detect_result.parent, ignore_errors=True)


def block_double_call(
        result: np.ndarray | Path | None,
        first_call_flag: bool,
        double_call_flag: bool,
    ) -> tuple[np.ndarray | Path | None, bool, bool]:
    if first_call_flag:
        return result, True, True
    return result, True, False


def reset_flags() -> tuple[bool, bool]:
    return False, False


def show_results(csv_path: Path) -> MatplotlibFigure | None:
    if csv_path is not None:
        return get_matplotlib_fig(csv_path)
    else:
        gr.Info(gr_info)


# =================== INTERFACE COMPONENTS ============================


def get_output_media_components(
        detect_result: np.ndarray | Path | None = None,
        double_call_flag: bool = False,
    ) -> tuple[gr.Image, gr.Video, gr.Button]:

    if double_call_flag:
        return gr.skip(), gr.skip(), gr.skip()

    visible = isinstance(detect_result, np.ndarray)
    image_output = gr.Image(
        value=detect_result if visible else None,
        type='numpy',
        width=800,
        height=636, # 640, 680, 720
        visible=visible,
        label='Output',
    )
    visible = isinstance(detect_result, Path)
    video_output = gr.Video(
        value=detect_result if visible else None,
        width=800,
        height=636,
        visible=visible,
        label='Output',
    )
    clear_btn = gr.Button(
        value='Clear Results',
        visible=detect_result is not None,
    )
    return image_output, video_output, clear_btn


def get_tracker_name_components(detect_mode: str) -> tuple[gr.Radio, gr.Checkbox]:
    visible = detect_mode == 'Tracking'
    tracker_name = gr.Radio(
        choices=CONFIG.TRACKER_NAMES,
        value=CONFIG.TRACKER_NAMES[0],
        label='Tracker Name',
        visible=visible,
        scale=3,
        )
    reinit_tracker = gr.Checkbox(
        value=False,
        label='Re-init tracker',
        info='Re-initialize the tracker on each detection run',
        visible=visible,
        scale=1,
        )
    return tracker_name, reinit_tracker

    
def get_download_csv_btn(csv_annotations_path: Path | None = None) -> gr.DownloadButton:
    download_csv_btn = gr.DownloadButton(
        label='Download csv annotations for video',
        value=csv_annotations_path,
        visible=csv_annotations_path is not None,
        )
    return download_csv_btn


# =================== APP INTERFACE ==========================

css = '''
.gradio-container {
    width:85% !important;
    margin: 0 auto !important;
}
.webcam-group {
    max-width: 600px !important; 
    max-height: none !important;
}
.webcam-column {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
'''

with gr.Blocks(css=css) as demo:
    model_state = gr.State({'model': default_model})
    
    with gr.Tab('Image/video detection'):
        detect_result = gr.State(None)
        csv_annotations_path = gr.State(None)
        first_call_flag = gr.State(False)
        double_call_flag = gr.State(False)
        
        with gr.Row():
            with gr.Column():
                file_path = gr.File(
                    file_types=['image', 'video'],
                    file_count='single',
                    label='Select image or video',
                    height=110,
                )
                file_link = gr.Textbox(label='Direct link to image or YouTube link')
                with gr.Row():
                    model_name = gr.Radio(choices=CONFIG.MODEL_NAMES, value=CONFIG.MODEL_NAMES[0], label='YOLO model')
                with gr.Row():
                    detect_mode = gr.Radio(
                        choices=CONFIG.DETECT_MODE_NAMES,
                        value=CONFIG.DETECT_MODE_NAMES[0],
                        label='Detect Mode',
                        scale=3,
                    )
                    tracker_name, reinit_tracker = get_tracker_name_components(detect_mode.value)

                with gr.Group():
                    conf = gr.Slider(0, 1, value=0.5, step=0.01, label='Confidence threshold')
                    iou = gr.Slider(0, 1, value=0.7, step=0.01, label='IOU threshold')
                with gr.Group():
                    status_message = gr.Textbox(value=get_ready_status(), label='Status', interactive=False)
                detect_btn = gr.Button('Detect')
                stop_btn = gr.Button('Stop detection and save result video', visible=False)

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
            show_progress='hidden',
        ).success(
            fn=lambda: gr.update(interactive=True),
            inputs=None,
            outputs=[detect_btn],
        )

        detect_mode.change(
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=[detect_btn],
        ).then(
            fn=get_tracker_name_components,
            inputs=[detect_mode],
            outputs=[tracker_name, reinit_tracker],
        ).then(
            fn=change_model,
            inputs=[model_state, model_name],
            outputs=[status_message],
            show_progress='hidden',
        ).then(
            fn=lambda: gr.update(interactive=True),
            inputs=None,
            outputs=[detect_btn],
        ).then(
            fn=get_ready_status,
            inputs=None,
            outputs=[status_message],
        )
        
        tracker_name.change(
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=[detect_btn],
        ).then(
            fn=change_model,
            inputs=[model_state, model_name],
            outputs=[status_message],
            show_progress='hidden',
        ).then(
            fn=lambda: gr.update(interactive=True),
            inputs=None,
            outputs=[detect_btn],
        ).then(
            fn=get_ready_status,
            inputs=None,
            outputs=[status_message],
        )

        detect_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[stop_btn],
            queue=False,
        )
        detect_btn.click(
            fn=reset_flags,
            inputs=None,
            outputs=[first_call_flag, double_call_flag],
            queue=False,
        )
        detect_btn.click(
            fn=reinit_model,
            inputs=[model_state, model_name, reinit_tracker],
            outputs=None,
            queue=False,
        )

        detect_event = detect_btn.click(
            fn=detect,
            inputs=[file_path, file_link, model_state, conf, iou, detect_mode, tracker_name],
            outputs=[detect_result, status_message],
            queue=True,
        )
        detect_event.then(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=[stop_btn],
            queue=False,
        )
        double_call_event = detect_event.success(
            fn=block_double_call,
            inputs=[detect_result, first_call_flag, double_call_flag],
            outputs=[detect_result, first_call_flag, double_call_flag],
        )
        double_call_event.success(
            fn=get_video_path_from_ultralytics_result,
            inputs=[detect_result],
            outputs=[detect_result],
        ).then(
            fn=update_detect_status,
            inputs=[detect_result, status_message],
            outputs=[status_message],
        ).then(
            fn=get_output_media_components,
            inputs=[detect_result, double_call_flag],
            outputs=[image_output, video_output, clear_btn],
        ).then(
            fn=get_csv_annotate,
            inputs=[detect_result, csv_annotations_path, double_call_flag],
            outputs=[csv_annotations_path],
        ).success(
            fn=get_download_csv_btn,
            inputs=[csv_annotations_path],
            outputs=[download_csv_btn],
            queue=False,
        ).then(
            fn=get_ready_status,
            inputs=None,
            outputs=[status_message],
        )

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[detect_event],
            queue=False,
        )

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
        <a href="https://github.com/sergey21000/yolo-detector" target='_blank'>GitHub Repository</a></h3>
        """)


    with gr.Tab('Show detection video results'):
        show_results_btn = gr.Button('Show detection results', scale=1)
        gr_info = 'To display the results, perform video detection on the first tab'
        show_results_btn.click(
            fn=show_results,
            inputs=[csv_annotations_path],
            outputs=gr.Plot(),
        )

    with gr.Tab('Webcam detection'):
        with gr.Column(elem_classes=['webcam-column']):
            with gr.Group(elem_classes=['webcam-group']):
                webcam = WebRTC(
                    label='Webcam Stream',
                    height=None,
                    width=None,
                    mode='send-receive',
                    modality='video',
                    mirror_webcam=True,
                    rtp_params={'degradationPreference': 'maintain-resolution'},
                )
                with gr.Column():
                    with gr.Row():
                        detect_mode = gr.Radio(
                            choices=CONFIG.DETECT_MODE_NAMES,
                            value=CONFIG.DETECT_MODE_NAMES[0],
                            label='Detect Mode',
                        )
                        tracker_name = gr.Radio(
                            choices=CONFIG.TRACKER_NAMES,
                            value=CONFIG.TRACKER_NAMES[0],
                            label='Tracker Name',
                            visible=detect_mode.value == 'Tracking',
                        )
                    conf = gr.Slider(0, 1, value=0.5, step=0.05, label='Confidence')
                    iou = gr.Slider(0, 1, value=0.7, step=0.05, label='IOU')
                    
        detect_mode.change(
            fn=lambda mode: gr.update(visible=True) if mode == 'Tracking' else gr.skip(),
            inputs=[detect_mode],
            outputs=[tracker_name],
        ).then(
            fn=change_model,
            inputs=[model_state, model_name],
            outputs=[status_message],
        )
        tracker_name.change(
            fn=change_model,
            inputs=[model_state, model_name],
            outputs=[status_message],
        )
        webcam.stream(
            fn=detect_webcam,
            inputs=[webcam, model_state, conf, iou, detect_mode, tracker_name],
            outputs=[webcam],
            time_limit=CONFIG.WEBCAM_TIME_LIMIT,
        )

if __name__ == '__main__':
    demo.launch()