import time
import os
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tracemalloc  # 메모리 추적
from statistics import mean, stdev
import matplotlib.pyplot as plt
import psutil

# 모델 경로 설정
save_dir = Path(r"/home/jetson/Downloads/yolov/yolov8m/yolov8m_saved_model/")
tflite_float16_path = save_dir / "yolov8m_float16.tflite"
tflite_float32_path = save_dir / "yolov8m_float32.tflite"
onnx_model_path = "yolov8m.onnx"


# 데이터셋 경로 설정
dataset_path = "coco128/images/train2017"

def load_interpreters():
    # TFLite Interpreter 설정 (float16 및 float32)
    tf_lite_interpreter_16 = tf.lite.Interpreter(model_path=str(tflite_float16_path))
    tf_lite_interpreter_16.allocate_tensors()