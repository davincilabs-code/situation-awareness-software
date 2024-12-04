# import time
# import os
# import numpy as np
# from ultralytics import YOLO
# import tensorflow as tf
# import onnxruntime as ort
# from pathlib import Path
# from tqdm import tqdm
# from PIL import Image
# import requests
# import zipfile

# # 데이터셋 다운로드 및 경로 설정
# dataset_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
# dataset_path = "coco128"

# # 다운로드와 압축 해제를 파이썬 코드로 대체
# def download_and_extract(url, dest_path):
#     zip_path = "coco128.zip"
#     # 파일 다운로드
#     if not os.path.exists(dest_path):
#         print("Downloading dataset...")
#         response = requests.get(url)
#         with open(zip_path, "wb") as f:
#             f.write(response.content)
#         # 압축 해제
#         print("Extracting dataset...")
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall()
#         os.remove(zip_path)  # 압축 파일 삭제

# download_and_extract(dataset_url, dataset_path)

# # 모델 경로 설정 (YOLOv8n.pt 파일의 경로로 변경하세요)
# model_path = "yolov8n.pt"

# def load_models(model_path):
#     """TensorFlow Lite와 ONNX 모델을 로드합니다."""
#     # TensorFlow Lite 모델 로드
#     tf_lite_model = YOLO(model_path)
#     tf_lite_model.export(format='tflite')
#     tflite_path = str(Path(model_path).with_suffix('.tflite'))
#     tf_lite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
#     tf_lite_interpreter.allocate_tensors()
    
#     # ONNX 모델 로드
#     onnx_path = str(Path(model_path).with_suffix('.onnx'))
#     YOLO(model_path).export(format='onnx')
#     ort_sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    
#     return tf_lite_interpreter, ort_sess

# def prepare_image(image_path, input_size=(640, 640)):
#     """이미지를 전처리합니다."""
#     img = Image.open(image_path)
#     img = img.resize(input_size)
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0).astype(np.float32)
#     return img

# def benchmark_inference(model_tf_lite, model_onnx, image_paths, num_iterations=100):
#     """추론 성능을 벤치마크합니다."""
#     # TensorFlow Lite 벤치마크
#     tf_lite_times = []
#     for image_path in tqdm(image_paths, desc="TensorFlow Lite"):
#         image = prepare_image(image_path)
#         for _ in range(num_iterations):
#             start_time = time.perf_counter()
#             model_tf_lite.set_tensor(model_tf_lite.get_input_details()[0]['index'], image)
#             model_tf_lite.invoke()
#             _ = model_tf_lite.get_tensor(model_tf_lite.get_output_details()[0]['index'])
#             tf_lite_times.append(time.perf_counter() - start_time)
    
#     # ONNX 벤치마크
#     onnx_times = []
#     input_name = model_onnx.get_inputs()[0].name
#     for image_path in tqdm(image_paths, desc="ONNX"):
#         image = prepare_image(image_path)
#         for _ in range(num_iterations):
#             start_time = time.perf_counter()
#             _ = model_onnx.run(None, {input_name: image})
#             onnx_times.append(time.perf_counter() - start_time)
    
#     return tf_lite_times, onnx_times

# def main():
#     # 데이터셋 이미지 경로 리스트 생성
#     image_folder = Path(dataset_path) / "images" / "train"
#     image_paths = [str(p) for p in image_folder.glob("*.jpg")]
    
#     print("모델 로딩 중...")
#     tf_lite_model, onnx_model = load_models(model_path)
    
#     print("벤치마크 시작...")
#     tf_lite_times, onnx_times = benchmark_inference(tf_lite_model, onnx_model, image_paths)
    
#     # 결과 출력
#     print("\n=== 벤치마크 결과 ===")
#     print(f"TensorFlow Lite 평균 추론 시간: {np.mean(tf_lite_times[10:]) * 1000:.2f}ms")
#     print(f"TensorFlow Lite 표준 편차: {np.std(tf_lite_times[10:]) * 1000:.2f}ms")
#     print(f"ONNX 평균 추론 시간: {np.mean(onnx_times[10:]) * 1000:.2f}ms")
#     print(f"ONNX 표준 편차: {np.std(onnx_times[10:]) * 1000:.2f}ms")
#     print(f"ONNX vs TF Lite 속도 비율: {np.mean(tf_lite_times[10:]) / np.mean(onnx_times[10:]):.2f}x")

# if __name__ == "__main__":
#     main()

import time
import os
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import requests
import zipfile

# 데이터셋 다운로드 및 경로 설정
dataset_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
dataset_path = "coco128"

def download_and_extract(url, dest_path):
    zip_path = "coco128.zip"
    if not os.path.exists(dest_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)  # 압축 파일 삭제

download_and_extract(dataset_url, dataset_path)

model_path = "yolov8m.pt"

def load_models(model_path):
    tf_lite_model = YOLO(model_path)
    tf_lite_model.export(format='tflite')

    tflite_path = Path(model_path).with_suffix('.tflite')

    if not tflite_path.exists():
        print(f"Error: {tflite_path} not found. Please check the export process.")
        return None, None
    
    tf_lite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    tf_lite_interpreter.allocate_tensors()

    onnx_path = Path(model_path).with_suffix('.onnx')
    YOLO(model_path).export(format='onnx')
    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found. Please check the export process.")
        return None, None
    
    ort_sess = ort.InferenceSession(str(onnx_path), providers=['CUDAExecutionProvider'])

    return tf_lite_interpreter, ort_sess

def prepare_image(image_path, input_size=(640, 640)):
    img = Image.open(image_path)
    img = img.resize(input_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def benchmark_inference(model_tf_lite, model_onnx, image_paths, num_iterations=100):
# import time
# import os
# import numpy as np
# from ultralytics import YOLO
# import tensorflow as tf
# import onnxruntime as ort
# from pathlib import Path
# from tqdm import tqdm
# from PIL import Image
# import requests
# import zipfile
# import shutil
# from tensorflow.lite.python import lite

# # 데이터셋 다운로드 및 경로 설정
# dataset_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
# dataset_path = "coco128"

# def download_and_extract(url, dest_path):
#     zip_path = "coco128.zip"
#     if not os.path.exists(dest_path):
#         print("Downloading dataset...")
#         response = requests.get(url)
#         with open(zip_path, "wb") as f:
#             f.write(response.content)
#         print("Extracting dataset...")
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall()
#         os.remove(zip_path)

# download_and_extract(dataset_url, dataset_path)

# # 모델 경로 설정 및 저장 디렉토리 지정
# model_path = "yolov8n.pt"
# save_dir = Path("C:/Users/USER/Desktop/2024/프로젝트/yolov/yolov8n_saved_model")

# def load_models(model_path):
#     """TensorFlow Lite (float16, float32)와 ONNX 모델을 로드합니다."""
#     yolo_model = YOLO(model_path)
    
#     # TensorFlow Lite float32 모델 내보내기
#     yolo_model.export(format='tflite')
#     default_tflite_32_path = Path(model_path).with_suffix('.tflite')
#     target_tflite_32_path = save_dir / "yolov8n_float32.tflite"
#     if default_tflite_32_path.exists():
#         save_dir.mkdir(parents=True, exist_ok=True)
#         shutil.move(str(default_tflite_32_path), str(target_tflite_32_path))
#     else:
#         print("Error: TFLite float32 model export failed.")
#         return None, None, None

#     # TensorFlow Lite float16 양자화
#     converter = lite.TFLiteConverter.from_saved_model(str(target_tflite_32_path))
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_types = [tf.float16]
#     tflite_float16_model = converter.convert()
    
#     # float16 모델 저장
#     target_tflite_16_path = save_dir / "yolov8n_float16.tflite"
#     with open(target_tflite_16_path, "wb") as f:
#         f.write(tflite_float16_model)

#     # TensorFlow Lite 해석기 초기화
#     tf_lite_interpreter_16 = tf.lite.Interpreter(model_path=str(target_tflite_16_path))
#     tf_lite_interpreter_16.allocate_tensors()
#     tf_lite_interpreter_32 = tf.lite.Interpreter(model_path=str(target_tflite_32_path))
#     tf_lite_interpreter_32.allocate_tensors()
    
#     # ONNX 모델 내보내기 및 경로 설정
#     yolo_model.export(format='onnx')
#     default_onnx_path = Path(model_path).with_suffix('.onnx')
#     target_onnx_path = save_dir / "yolov8n.onnx"
#     if default_onnx_path.exists():
#         shutil.move(str(default_onnx_path), str(target_onnx_path))
#     else:
#         print("Error: ONNX model export failed.")
#         return None, None, None

#     ort_sess = ort.InferenceSession(str(target_onnx_path), providers=['CUDAExecutionProvider'])
    
#     return tf_lite_interpreter_16, tf_lite_interpreter_32, ort_sess

# def prepare_image(image_path, input_size=(640, 640)):
#     img = Image.open(image_path)
#     img = img.resize(input_size)
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0).astype(np.float32)
#     return img

# def benchmark_inference(model_tf_lite_16, model_tf_lite_32, model_onnx, image_paths, num_iterations=100):
#     tf_lite_16_times, tf_lite_32_times, onnx_times = [], [], []
    
#     if model_tf_lite_16:
#         for image_path in tqdm(image_paths, desc="TensorFlow Lite Float16"):
#             image = prepare_image(image_path)
#             for _ in range(num_iterations):
#                 start_time = time.perf_counter()
#                 model_tf_lite_16.set_tensor(model_tf_lite_16.get_input_details()[0]['index'], image)
#                 model_tf_lite_16.invoke()
#                 _ = model_tf_lite_16.get_tensor(model_tf_lite_16.get_output_details()[0]['index'])
#                 tf_lite_16_times.append(time.perf_counter() - start_time)
    
#     if model_tf_lite_32:
#         for image_path in tqdm(image_paths, desc="TensorFlow Lite Float32"):
#             image = prepare_image(image_path)
#             for _ in range(num_iterations):
#                 start_time = time.perf_counter()
#                 model_tf_lite_32.set_tensor(model_tf_lite_32.get_input_details()[0]['index'], image)
#                 model_tf_lite_32.invoke()
#                 _ = model_tf_lite_32.get_tensor(model_tf_lite_32.get_output_details()[0]['index'])
#                 tf_lite_32_times.append(time.perf_counter() - start_time)
    
#     if model_onnx:
#         input_name = model_onnx.get_inputs()[0].name
#         for image_path in tqdm(image_paths, desc="ONNX"):
#             image = prepare_image(image_path)
#             for _ in range(num_iterations):
#                 start_time = time.perf_counter()
#                 _ = model_onnx.run(None, {input_name: image})
#                 onnx_times.append(time.perf_counter() - start_time)
    
#     return tf_lite_16_times, tf_lite_32_times, onnx_times

# def main():
#     image_folder = Path(dataset_path) / "images" / "train"
#     image_paths = [str(p) for p in image_folder.glob("*.jpg")]
    
#     print("모델 로딩 중...")
#     tf_lite_model_16, tf_lite_model_32, onnx_model = load_models(model_path)
    
#     if tf_lite_model_16 is None or tf_lite_model_32 is None or onnx_model is None:
#         print("모델 로드에 실패했습니다.")
#         return
    
#     print("벤치마크 시작...")
#     tf_lite_16_times, tf_lite_32_times, onnx_times = benchmark_inference(
#         tf_lite_model_16, tf_lite_model_32, onnx_model, image_paths)
    
#     print("\n=== 벤치마크 결과 ===")
#     if tf_lite_16_times:
#         print(f"TensorFlow Lite Float16 평균 추론 시간: {np.mean(tf_lite_16_times[10:]) * 1000:.2f}ms")
#         print(f"TensorFlow Lite Float16 표준 편차: {np.std(tf_lite_16_times[10:]) * 1000:.2f}ms")
#     if tf_lite_32_times:
#         print(f"TensorFlow Lite Float32 평균 추론 시간: {np.mean(tf_lite_32_times[10:]) * 1000:.2f}ms")
#         print(f"TensorFlow Lite Float32 표준 편차: {np.std(tf_lite_32_times[10:]) * 1000:.2f}ms")
#     if onnx_times:
#         print(f"ONNX 평균 추론 시간: {np.mean(onnx_times[10:]) * 1000:.2f}ms")
#         print(f"ONNX 표준 편차: {np.std(onnx_times[10:]) * 1000:.2f}ms")

# if __name__ == "__main__":
#     main()
