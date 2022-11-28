# Import Libraries
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from tkinter import * 
from tkinter.ttk import *

import argparse
import atexit
import copy
import logging
import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from tkinter import * 
from tkinter.ttk import *

import cv2 
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

from mmaction.models import build_detector

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 webcam spatio-temporal detection demo')

    parser.add_argument(
        '--config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.5,
        help='the threshold of human action score')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--input-video',
        default='0',
        type=str,
        help='webcam id or input video file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/ava/label_map.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--output-fps',
        default= 60,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--out-filename',
        default='demo/demo_out.mp4',
        type=str,
        help='the filename of output video')
    parser.add_argument(
        '--show',
        default= True,
        action='store_true',
        help='Whether to show results with cv2.imshow')
    parser.add_argument(
        '--display-height',
        type=int,
        default=Tk().winfo_screenheight(),
        help='Image height for human detector and draw frames.')
    parser.add_argument(
        '--display-width',
        type=int,
        default=Tk().winfo_screenwidth(),
        help='Image width for human detector and draw frames.')
    parser.add_argument(
        '--predict-stepsize',
        default=60,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument(
        '--clip-vis-length',
        default=60,
        type=int,
        help='Number of draw frames per clip.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()
    return args


class TaskInfo:
    """Wapper for a clip.

    Transmit data around three threads.

    1) Read Thread: Create task and put task into read queue. Init `frames`,
        `processed_frames`, `img_shape`, `ratio`, `clip_vis_length`.
    2) Main Thread: Get data from read queue, predict human bboxes and stdet
        action labels, draw predictions and put task into display queue. Init
        `display_bboxes`, `stdet_bboxes` and `action_preds`, update `frames`.
    3) Display Thread: Get data from display queue, show/write frames and
        delete task.
    """

    def __init__(self):
        self.id = -1

        # raw frames, used as human detector input, draw predictions input
        # and output, display input
        self.frames = None

        # stdet params
        self.processed_frames = None  # model inputs
        self.frames_inds = None  # select frames from processed frames
        self.img_shape = None  # model inputs, processed frame shape
        # `action_preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        self.action_preds = None  # stdet results

        # human bboxes with the format (xmin, ymin, xmax, ymax)
        self.display_bboxes = None  # bboxes coords for self.frames
        self.stdet_bboxes = None  # bboxes coords for self.processed_frames
        self.ratio = None  # processed_frames.shape[1::-1]/frames.shape[1::-1]

        # for each clip, draw predictions on clip_vis_length frames
        self.clip_vis_length = -1

    def add_frames(self, idx, frames, processed_frames):
        """Add the clip and corresponding id.

        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
            processed_frames (list[ndarray]): list of resize and normed images
                in "BGR" format.
        """
        self.frames = frames
        self.processed_frames = processed_frames
        self.id = idx
        self.img_shape = processed_frames[0].shape[:2]

    def add_bboxes(self, display_bboxes):
        """Add correspondding bounding boxes."""
        self.display_bboxes = display_bboxes
        self.stdet_bboxes = display_bboxes.clone()
        self.stdet_bboxes[:, ::2] = self.stdet_bboxes[:, ::2] * self.ratio[0]
        self.stdet_bboxes[:, 1::2] = self.stdet_bboxes[:, 1::2] * self.ratio[1]

    def add_action_preds(self, preds):
        """Add the corresponding action predictions."""
        self.action_preds = preds

    def get_model_inputs(self, device):
        """Convert preprocessed images to MMAction2 STDet model inputs."""
        cur_frames = [self.processed_frames[idx] for idx in self.frames_inds]
        input_array = np.stack(cur_frames).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(device)
        return dict(
            return_loss=False,
            img=[input_tensor],
            proposals=[[self.stdet_bboxes]],
            img_metas=[[dict(img_shape=self.img_shape)]])


class BaseHumanDetector(metaclass=ABCMeta):
    """Base class for Human Dector.

    Args:
        device (str): CPU/CUDA device option.
    """

    def __init__(self, device):
        self.device = torch.device(device)

    @abstractmethod
    def _do_detect(self, image):
        """Get human bboxes with shape [n, 4].

        The format of bboxes is (xmin, ymin, xmax, ymax) in pixels.
        """

    def predict(self, task):
        """Add keyframe bboxes to task."""
        # keyframe idx == (clip_len * frame_interval) // 2
        keyframe = task.frames[len(task.frames) // 2]

        # call detector
        bboxes = self._do_detect(keyframe)

        # convert bboxes to torch.Tensor and move to target device
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(self.device)
        elif isinstance(bboxes, torch.Tensor) and bboxes.device != self.device:
            bboxes = bboxes.to(self.device)

        # update task
        task.add_bboxes(bboxes)

        return task


class MmdetHumanDetector(BaseHumanDetector):
    """Wrapper for mmdetection human detector.

    Args:
        config (str): Path to mmdetection config.
        ckpt (str): Path to mmdetection checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human detection score.
        person_classid (int): Choose class from detection results.
            Default: 0. Suitable for COCO pretrained models.
    """

    def __init__(self, config, ckpt, device, score_thr, person_classid=0):
        super().__init__(device)
        self.model = init_detector(config, ckpt, device)
        self.person_classid = person_classid
        self.score_thr = score_thr

    def _do_detect(self, image):
        """Get bboxes in shape [n, 4] and values in pixels."""
        result = inference_detector(self.model, image)[self.person_classid]
        result = result[result[:, 4] >= self.score_thr][:, :4]
        return result


class StdetPredictor:
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        config (str): Path to stdet config.
        ckpt (str): Path to stdet checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human action score.
        label_map_path (str): Path to label map file. The format for each line
            is `{class_id}: {class_name}`.
    """

    def __init__(self, config, checkpoint, device, score_thr, label_map_path):
        self.score_thr = score_thr

        # load model
        config.model.backbone.pretrained = None
        model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location='cpu')
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

        # init label map, aka class_id to class_name dict
        with open(label_map_path) as f:
            lines = f.readlines()
        lines = [x.strip().split(': ') for x in lines]
        self.label_map = {int(x[0]): x[1] for x in lines}
        try:
            if config['data']['train']['custom_classes'] is not None:
                self.label_map = {
                    id + 1: self.label_map[cls]
                    for id, cls in enumerate(config['data']['train']
                                             ['custom_classes'])
                }
        except KeyError:
            pass

    def predict(self, task):
        """Spatio-temporval Action Detection model inference."""
        # No need to do inference if no one in keyframe
        if len(task.stdet_bboxes) == 0:
            return task

        with torch.no_grad():
            result = self.model(**task.get_model_inputs(self.device))[0]

        # pack results of human detector and stdet
        preds = []
        for _ in range(task.stdet_bboxes.shape[0]):
            preds.append([])
        for class_id in range(len(result)):
            if class_id + 1 not in self.label_map:
                continue
            for bbox_id in range(task.stdet_bboxes.shape[0]):
                if result[class_id][bbox_id, 4] > self.score_thr:
                    preds[bbox_id].append((self.label_map[class_id + 1],
                                           result[class_id][bbox_id, 4]))

        # update task
        # `preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        task.add_action_preds(preds)

        return task


class ClipHelper:
    """Multithrading utils to manage the lifecycle of task."""

    def __init__(self,
                 config,
                 display_height=0,
                 display_width=0,
                 input_video=0,
                 predict_stepsize=40,
                 output_fps=25,
                 clip_vis_length=8,
                 out_filename=None,
                 show=True,
                 stdet_input_shortside=256):
        # stdet sampling strategy
        val_pipeline = config.data.val.pipeline
        sampler = [x for x in val_pipeline
                   if x['type'] == 'SampleAVAFrames'][0]
        clip_len, frame_interval = sampler['clip_len'], sampler[
            'frame_interval']
        self.window_size = clip_len * frame_interval

        # asserts
        assert (out_filename or show), \
            'out_filename and show cannot both be None'
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'
        assert clip_vis_length <= predict_stepsize
        assert 0 < predict_stepsize <= self.window_size

        # source params
        try:
            self.cap = cv2.VideoCapture(int(input_video))
            self.webcam = True
        except ValueError:
            self.cap = cv2.VideoCapture(input_video)
            self.webcam = False
        assert self.cap.isOpened()

        # stdet input preprocessing params
        h = int(self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height))
        w = int(self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width))
        self.stdet_input_size = mmcv.rescale_size(
            (w, h), (stdet_input_shortside, np.Inf))
        img_norm_cfg = config['img_norm_cfg']
        if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
            to_bgr = img_norm_cfg.pop('to_bgr')
            img_norm_cfg['to_rgb'] = to_bgr
        img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
        img_norm_cfg['std'] = np.array(img_norm_cfg['std'])
        self.img_norm_cfg = img_norm_cfg

        # task init params
        self.clip_vis_length = clip_vis_length
        self.predict_stepsize = predict_stepsize
        self.buffer_size = self.window_size - self.predict_stepsize
        frame_start = self.window_size // 2 - (clip_len // 2) * frame_interval
        self.frames_inds = [
            frame_start + frame_interval * i for i in range(clip_len)
        ]
        self.buffer = []
        self.processed_buffer = []

        # output/display params
        if display_height > 0 and display_width > 0:
            self.display_size = (display_width, display_height)
        elif display_height > 0 or display_width > 0:
            self.display_size = mmcv.rescale_size(
                (w, h), (np.Inf, max(display_height, display_width)))
        else:
            self.display_size = (w, h)
        self.ratio = tuple(
            n / o for n, o in zip(self.stdet_input_size, self.display_size))
        if output_fps <= 0:
            self.output_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        else:
            self.output_fps = output_fps
        self.show = show
        self.video_writer = None
        if out_filename is not None:
            self.video_writer = self.get_output_video_writer(out_filename)
        display_start_idx = self.window_size // 2 - self.predict_stepsize // 2
        self.display_inds = [
            display_start_idx + i for i in range(self.predict_stepsize)
        ]

        # display multi-theading params
        self.display_id = -1  # task.id for display queue
        self.display_queue = {}
        self.display_lock = threading.Lock()
        self.output_lock = threading.Lock()

        # read multi-theading params
        self.read_id = -1  # task.id for read queue
        self.read_id_lock = threading.Lock()
        self.read_queue = queue.Queue()
        self.read_lock = threading.Lock()
        self.not_end = True  # cap.read() flag

        # program state
        self.stopped = False

        atexit.register(self.clean)

    def read_fn(self):
        """Main function for read thread.

        Contains three steps:

        1) Read and preprocess (resize + norm) frames from source.
        2) Create task by frames from previous step and buffer.
        3) Put task into read queue.
        """
        was_read = True
        start_time = time.time()
        while was_read and not self.stopped:
            # init task
            task = TaskInfo()
            task.clip_vis_length = self.clip_vis_length
            task.frames_inds = self.frames_inds
            task.ratio = self.ratio

            # read buffer
            frames = []
            processed_frames = []
            if len(self.buffer) != 0:
                frames = self.buffer
            if len(self.processed_buffer) != 0:
                processed_frames = self.processed_buffer

            # read and preprocess frames from source and update task
            with self.read_lock:
                before_read = time.time()
                read_frame_cnt = self.window_size - len(frames)
                while was_read and len(frames) < self.window_size:
                    was_read, frame = self.cap.read()
                    frame = cv2.flip(frame, 1)
                    if not self.webcam:
                        # Reading frames too fast may lead to unexpected
                        # performance degradation. If you have enough
                        # resource, this line could be commented.
                        time.sleep(1 / self.output_fps)
                    if was_read:
                        frames.append(mmcv.imresize(frame, self.display_size))
                        processed_frame = mmcv.imresize(
                            frame, self.stdet_input_size).astype(np.float32)
                        _ = mmcv.imnormalize_(processed_frame,
                                              **self.img_norm_cfg)
                        processed_frames.append(processed_frame)
            task.add_frames(self.read_id + 1, frames, processed_frames)

            # update buffer
            if was_read:
                self.buffer = frames[-self.buffer_size:]
                self.processed_buffer = processed_frames[-self.buffer_size:]

            # update read state
            with self.read_id_lock:
                self.read_id += 1
                self.not_end = was_read

            self.read_queue.put((was_read, copy.deepcopy(task)))

    def display_fn(self):
        """Main function for display thread.

        Read data from display queue and display predictions.
        """
        start_time = time.time()
        while not self.stopped:
            # get the state of the read thread
            with self.read_id_lock:
                read_id = self.read_id
                not_end = self.not_end

            with self.display_lock:
                # If video ended and we have display all frames.
                if not not_end and self.display_id == read_id:
                    break

                # If the next task are not available, wait.
                if (len(self.display_queue) == 0 or
                        self.display_queue.get(self.display_id + 1) is None):
                    time.sleep(0.02)
                    continue

                # get display data and update state
                self.display_id += 1
                was_read, task = self.display_queue[self.display_id]
                del self.display_queue[self.display_id]
                display_id = self.display_id

            # do display predictions
            with self.output_lock:
                if was_read and task.id == 0:
                    # the first task
                    cur_display_inds = range(self.display_inds[-1] + 1)
                elif not was_read:
                    # the last task
                    cur_display_inds = range(self.display_inds[0],
                                             len(task.frames))
                else:
                    cur_display_inds = self.display_inds

                for frame_id in cur_display_inds: 
                    frame = task.frames[frame_id]
                    if self.show:
                        cv2.imshow('Spatiotemporal Action Detection', frame)
                        key = cv2.waitKey(int(1000 / self.output_fps))
                        if key == 27:  # ESC
                            self.stopped = True
                    if self.video_writer:
                        self.video_writer.write(frame)


    def __iter__(self):
        return self

    def __next__(self):
        """Get data from read queue.

        This function is part of the main thread.
        """
        if self.read_queue.qsize() == 0:
            time.sleep(0.02)
            return not self.stopped, None

        was_read, task = self.read_queue.get()
        if not was_read:
            # If we reach the end of the video, there aren't enough frames
            # in the task.processed_frames, so no need to model inference
            # and draw predictions. Put task into display queue.
            with self.read_id_lock:
                read_id = self.read_id
            with self.display_lock:
                self.display_queue[read_id] = was_read, copy.deepcopy(task)

            # main thread doesn't need to handle this task again
            task = None
        return was_read, task

    def start(self):
        """Start read thread and display thread."""
        self.read_thread = threading.Thread(
            target=self.read_fn, args=(), name='VidRead-Thread', daemon=True)
        self.read_thread.start()
        self.display_thread = threading.Thread(
            target=self.display_fn,
            args=(),
            name='VidDisplay-Thread',
            daemon=True)
        self.display_thread.start()

        return self

    def clean(self):
        """Close all threads and release all resources."""
        self.stopped = True
        self.read_lock.acquire()
        self.cap.release()
        self.read_lock.release()
        self.output_lock.acquire()
        cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()
        self.output_lock.release()

    def join(self):
        """Waiting for the finalization of read and display thread."""
        self.read_thread.join()
        self.display_thread.join()

    def display(self, task):
        """Add the visualized task to the display queue.

        Args:
            task (TaskInfo object): task object that contain the necessary
            information for prediction visualization.
        """
        with self.display_lock:
            self.display_queue[task.id] = (True, task)

    def get_output_video_writer(self, path):
        """Return a video writer object.

        Args:
            path (str): path to the output video file.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=float(self.output_fps),
            frameSize=self.display_size,
            isColor=True)


class BaseVisualizer(metaclass=ABCMeta):
    """Base class for visualization tools."""

    def __init__(self, max_labels_per_bbox):
        self.max_labels_per_bbox = max_labels_per_bbox

    def draw_predictions(self, task):
        """Visualize stdet predictions on raw frames."""
        # read bboxes from task
        bboxes = task.display_bboxes.cpu().numpy()

        # draw predictions and update task
        keyframe_idx = len(task.frames) // 2
        draw_range = [
            keyframe_idx - task.clip_vis_length // 2,
            keyframe_idx + (task.clip_vis_length - 1) // 2
        ]
        assert draw_range[0] >= 0 and draw_range[1] < len(task.frames)
        task.frames = self.draw_clip_range(task.frames, task.action_preds,
                                           bboxes, draw_range)

        return task

    def draw_clip_range(self, frames, preds, bboxes, draw_range):
        """Draw a range of frames with the same bboxes and predictions."""
        # no predictions to be draw
        if bboxes is None or len(bboxes) == 0:
            return frames

        # draw frames in `draw_range`
        left_frames = frames[:draw_range[0]]
        right_frames = frames[draw_range[1] + 1:]
        draw_frames = frames[draw_range[0]:draw_range[1] + 1]

        # get labels(texts) and draw predictions
        draw_frames = [
            self.draw_one_image(frame, bboxes, preds) for frame in draw_frames
        ]

        return list(left_frames) + draw_frames + list(right_frames)

    @abstractmethod
    def draw_one_image(self, frame, bboxes, preds):
        """Draw bboxes and corresponding texts on one frame."""

    @staticmethod
    def abbrev(name):
        """Get the abbreviation of label name:

        'take (an object) from (a person)' -> 'take ... from ...'
        """
        while name.find('(') != -1:
            st, ed = name.find('('), name.find(')')
            name = name[:st] + '...' + name[ed + 1:]
        return name


class DefaultVisualizer(BaseVisualizer):
    """Tools to visualize predictions.

    Args:
        max_labels_per_bbox (int): Max number of labels to visualize for a
            person box. Default: 5.
        plate (str): The color plate used for visualization. Two recommended
            plates are blue plate `03045e-023e8a-0077b6-0096c7-00b4d8-48cae4`
            and green plate `004b23-006400-007200-008000-38b000-70e000`. These
            plates are generated by https://coolors.co/.
            Default: '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'.
        text_fontface (int): Fontface from OpenCV for texts.
            Default: cv2.FONT_HERSHEY_DUPLEX.
        text_fontscale (float): Fontscale from OpenCV for texts.
            Default: 0.5.
        text_fontcolor (tuple): fontface from OpenCV for texts.
            Default: (255, 255, 255).
        text_thickness (int): Thickness from OpenCV for texts.
            Default: 1.
        text_linetype (int): LInetype from OpenCV for texts.
            Default: 1.
    """

    def __init__(
            self,
            max_labels_per_bbox=3,
            plate='03045e-023e8a-0077b6-0096c7-00b4d8-48cae4',
            text_fontface=cv2.FONT_HERSHEY_DUPLEX,
            text_fontscale=2,
            text_fontcolor=(255, 255, 255),  # white
            text_thickness=2,
            text_linetype=1):
        super().__init__(max_labels_per_bbox=max_labels_per_bbox)
        self.text_fontface = text_fontface
        self.text_fontscale = text_fontscale
        self.text_fontcolor = text_fontcolor
        self.text_thickness = text_thickness
        self.text_linetype = text_linetype

        def hex2color(h):
            """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
            return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

        plate = plate.split('-')
        self.plate = [hex2color(h) for h in plate]

    def draw_one_image(self, frame, bboxes, preds):
        """Draw predictions on one image."""
        for bbox, pred in zip(bboxes, preds):
            # draw bbox
            box = bbox.astype(np.int64)
            st, ed = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(frame, st, ed, (0, 0, 255), 2)

            # draw texts
            for k, (label, score) in enumerate(pred):
                if k >= self.max_labels_per_bbox:
                    break
                if self.abbrev(label) ==  'stand' or self.abbrev(label) == 'walk' or self.abbrev(label) == 'carry/hold ...':
                    text = f'{self.abbrev(label)}: {score:.4f}'
                    location = (5 +st[0],58 + k * 58 + st[1])
                    cv2.putText(frame, text, location,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10, cv2.LINE_AA)
                    cv2.putText(frame, text, location,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return frame


def main(args):

    # init action detector
    config = Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)

    # init clip helper
    clip_helper = ClipHelper(
        config=config,
        display_height=args.display_height,
        display_width=args.display_width,
        input_video=args.input_video,
        predict_stepsize=args.predict_stepsize,
        output_fps=args.output_fps,
        clip_vis_length=args.clip_vis_length,
        out_filename=args.out_filename,
        show=args.show)


    # init visualizer
    vis = DefaultVisualizer()

    # start read and display thread
    clip_helper.start()
    

    # init human detector
    human_detector = MmdetHumanDetector(args.det_config, args.det_checkpoint,
                                        args.device, args.det_score_thr)

    
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass
    
    stdet_predictor = StdetPredictor(
        config=config,
        checkpoint=args.checkpoint,
        device=args.device,
        score_thr=args.action_score_thr,
        label_map_path=args.label_map)

    # init gesture detector
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode= args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence= args.min_detection_confidence,
        min_tracking_confidence= args.min_tracking_confidence,
    )
    use_brect = True
    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    
      
    
    try:
        # Main thread main function contains:
        # 1) get data from read queue
        # 2) get human bboxes and stdet predictions
        # 3) draw stdet predictions and update task
        # 4) put task into display queue
        for able_to_read, task in clip_helper:
            # get data from read queue

            if not able_to_read:
                # read thread is dead and all tasks are processed
                break

            if task is None:
                # when no data in read queue, wait
                time.sleep(0.01)
                continue

            # get human bboxes
            human_detector.predict(task)

            # get stdet predictions
            stdet_predictor.predict(task)

            # draw stdet predictions in raw frames
            vis.draw_predictions(task)
    
            fps = cvFpsCalc.get()

            # select key mode 
            key = cv.waitKey(10)
            number, mode = select_mode(key, mode)

            for i, frame in enumerate(task.frames):
                
                frame.flags.writeable = False
                results = hands.process(frame)
                frame.flags.writeable = True

                # get hand gestures
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                        results.multi_handedness):
                        # Bounding box calculation
                        brect = calc_bounding_rect(frame, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(frame, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        pre_processed_point_history_list = pre_process_point_history(
                            frame, point_history)
                        # Write to the dataset file
                        logging_csv(number, mode, pre_processed_landmark_list,
                                    pre_processed_point_history_list)

                        # Hand sign classification
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        if hand_sign_id == 2:  # Point gesture
                            point_history.append(landmark_list[8])
                        else:
                            point_history.append([0, 0])

                        # Finger gesture classification
                        finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (history_length * 2):
                            finger_gesture_id = point_history_classifier(
                                pre_processed_point_history_list)

                        # Calculates the gesture IDs in the latest detection
                        finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            finger_gesture_history).most_common()
                
                        # Drawing part
                    frame = draw_bounding_rect(use_brect, frame, brect)
                    frame = draw_landmarks(frame, landmark_list) # Skeleton display
                    frame = draw_info_text(
                    frame,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                else:
                    point_history.append([0, 0])
                
                frame = draw_point_history(frame, point_history)
                frame = draw_info(frame, fps, mode, number)
                task.frames[i] = frame

            # add draw frames to display queue
            clip_helper.display(task)    

        # wait for display thread
        clip_helper.join()
    except KeyboardInterrupt:
        pass
    finally:
        # close read & display thread, release all resources
        clip_helper.clean()

# Gesture detection functions ########################################################
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 2)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = "Hand Gesture:" + hand_sign_text + '('+ info_text + ')' 
    cv.putText(image, info_text, (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10, cv.LINE_AA)
    cv.putText(image, info_text, (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
    if hand_sign_text == "Point":
        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 130),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 130),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                    cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (0, 0, 255), 10)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 1160), cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 1160), cv.FONT_HERSHEY_SIMPLEX,
               1, (255, 255, 255), 1, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main(parse_args())
