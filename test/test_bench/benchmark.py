# import time
# import os
# import numpy as np
# import onnxruntime as ort
# import tensorflow as tf
# from pathlib import Path
# from tqdm import tqdm
# from PIL import Image
# import tracemalloc  # 메모리 추적
# from statistics import mean, stdev
# import matplotlib.pyplot as plt
# import psutil

# # 모델 경로 설정
# save_dir = Path("/home/jetson/Downloads/yolov/yolov8n/yolov8n_saved_model/")
# tflite_float16_path = save_dir / "yolov8n_float16.tflite"
# tflite_float32_path = save_dir / "yolov8n_float32.tflite"
# onnx_model_path = "yolov8n.onnx"

# # 데이터셋 경로 설정
# dataset_path = "coco128/images/train2017"

# def load_interpreters():
#     """TensorFlow Lite와 ONNX 모델을 로드합니다."""
#     # TFLite Interpreter 설정 (float16 및 float32)
#     tf_lite_interpreter_16 = tf.lite.Interpreter(model_path=str(tflite_float16_path))
#     tf_lite_interpreter_16.allocate_tensors()
    
#     tf_lite_interpreter_32 = tf.lite.Interpreter(model_path=str(tflite_float32_path))
#     tf_lite_interpreter_32.allocate_tensors()
    
#     # ONNX Inference Session 설정 (CPU 사용)
#     ort_sess = ort.InferenceSession(str(onnx_model_path))  # providers 매개변수 제거
#     # ort_sess = ort.InferenceSession(str(onnx_model_path), providers=['CUDAExecutionProvider'])
    
#     return tf_lite_interpreter_16, tf_lite_interpreter_32, ort_sess

# def prepare_image_tflite(image_path, input_size=(640, 640)):
#     """TensorFlow Lite 모델 입력 형식으로 이미지를 준비합니다."""
#     img = Image.open(image_path).convert('RGB')  # 이미지가 RGB로 변환되었는지 확인
#     img = img.resize(input_size)                 # 모델 입력 크기로 리사이즈
#     img = np.array(img, dtype=np.float32) / 255.0  # 정규화하여 (height, width, channels)로 변환
#     img = np.expand_dims(img, axis=0)            # (1, 640, 640, 3) 형태로 만듦
#     return img

# def prepare_image_onnx(image_path, input_size=(640, 640)):
#     """ONNX 모델 입력 형식으로 이미지를 준비합니다."""
#     img = Image.open(image_path).convert('RGB')  # 항상 RGB로 변환하여 3채널을 보장
#     img = img.resize(input_size)                 # 모델 입력 크기로 리사이즈
#     img = np.array(img, dtype=np.float32) / 255.0  # 정규화 (H, W, C)
#     img = np.transpose(img, (2, 0, 1))           # (H, W, C) -> (C, H, W)
#     img = np.expand_dims(img, axis=0)            # (1, 3, 640, 640) 형태로 만듦
#     return img

# def benchmark_inference_extended(model_tf_lite_16, model_tf_lite_32, model_onnx, image_paths, num_iterations=1):
#     """각 모델의 추론 시간과 메모리 사용을 측정합니다."""
#     tf_lite_16_times, tf_lite_32_times, onnx_times = [], [], []
#     tf_lite_16_memories, tf_lite_32_memories, onnx_memories = [], [], []
    
#     # 현재 프로세스 정보 가져오기
#     process = psutil.Process(os.getpid())
    
#     # TensorFlow Lite Float16 모델
#     if model_tf_lite_16:
#         for image_path in tqdm(image_paths, desc="TensorFlow Lite Float16"):
#             image = prepare_image_tflite(image_path)
#             for _ in range(num_iterations):
#                 start_time = time.perf_counter()
#                 model_tf_lite_16.set_tensor(model_tf_lite_16.get_input_details()[0]['index'], image)
#                 model_tf_lite_16.invoke()
#                 tf_lite_16_times.append(time.perf_counter() - start_time)
#                 # 메모리 사용 측정
#                 memory_info = process.memory_info()
#                 tf_lite_16_memories.append(memory_info.rss)

#     # TensorFlow Lite Float32 모델
#     if model_tf_lite_32:
#         for image_path in tqdm(image_paths, desc="TensorFlow Lite Float32"):
#             image = prepare_image_tflite(image_path)
#             for _ in range(num_iterations):
#                 start_time = time.perf_counter()
#                 model_tf_lite_32.set_tensor(model_tf_lite_32.get_input_details()[0]['index'], image)
#                 model_tf_lite_32.invoke()
#                 tf_lite_32_times.append(time.perf_counter() - start_time)
#                 # 메모리 사용 측정
#                 memory_info = process.memory_info()
#                 tf_lite_32_memories.append(memory_info.rss)

#     # ONNX 모델
#     if model_onnx:
#         input_name = model_onnx.get_inputs()[0].name
#         for image_path in tqdm(image_paths, desc="ONNX"):
#             image = prepare_image_onnx(image_path)
#             for _ in range(num_iterations):
#                 start_time = time.perf_counter()
#                 model_onnx.run(None, {input_name: image})
#                 onnx_times.append(time.perf_counter() - start_time)
#                 # 메모리 사용 측정
#                 memory_info = process.memory_info()
#                 onnx_memories.append(memory_info.rss)
    
#     return tf_lite_16_times, tf_lite_32_times, onnx_times, tf_lite_16_memories, tf_lite_32_memories, onnx_memories

# def save_benchmark_graphs(tf_lite_16_times, tf_lite_32_times, onnx_times, 
#                           tf_lite_16_memories, tf_lite_32_memories, onnx_memories):
#     # 모델 명칭과 데이터를 정리
#     model_names = ["TFLite Float16", "TFLite Float32", "ONNX"]
#     avg_times = [mean(tf_lite_16_times[10:]) * 1000, mean(tf_lite_32_times[10:]) * 1000, mean(onnx_times[10:]) * 1000]
#     std_times = [stdev(tf_lite_16_times[10:]) * 1000, stdev(tf_lite_32_times[10:]) * 1000, stdev(onnx_times[10:]) * 1000]
#     avg_memories = [mean(tf_lite_16_memories[10:]) / 1024, mean(tf_lite_32_memories[10:]) / 1024, mean(onnx_memories[10:]) / 1024]

#     colors = ['skyblue', 'salmon', 'lightgreen']  # 색상 정의

#     # 추론 시간 그래프
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(model_names, avg_times, yerr=std_times, capsize=5, color=colors)
#     plt.ylabel("Average Inference Time (ms)")
#     plt.title("Benchmark: Average Inference Time")
#     for i, v in enumerate(avg_times):
#         plt.text(i, v + 5, f"{v:.2f} ms", ha='center', fontweight='bold')  # 각 막대 위에 값 라벨 추가

#     # 범례를 오른쪽 중앙에 추가
#     plt.legend(bars, model_names, loc='center right', title="Model Type")
#     plt.savefig("inference_time_benchmark.png")
#     plt.close()

#     # 메모리 사용량 그래프
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(model_names, avg_memories, color=colors)
#     plt.ylabel("Average Memory Usage (KB)")
#     plt.title("Benchmark: Average Memory Usage")
#     for i, v in enumerate(avg_memories):
#         plt.text(i, v + 5, f"{v:.2f} KB", ha='center', fontweight='bold')  # 각 막대 위에 값 라벨 추가

#     # 범례를 오른쪽 중앙에 추가
#     plt.legend(bars, model_names, loc='center right', title="Model Type")
#     plt.savefig("memory_usage_benchmark.png")
#     plt.close()

#     print("추론 시간 및 메모리 사용량 그래프가 저장되었습니다.")

# # main 함수 수정
# def main():
#     # 이미지 파일 목록 생성
#     image_folder = Path(dataset_path)
#     image_paths = [str(p) for p in image_folder.glob("*.jpg")]
    
#     print("모델 로딩 중...")
#     tf_lite_model_16, tf_lite_model_32, onnx_model = load_interpreters()
    
#     print("벤치마크 시작...")
#     tf_lite_16_times, tf_lite_32_times, onnx_times, tf_lite_16_memories, tf_lite_32_memories, onnx_memories = benchmark_inference_extended(
#         tf_lite_model_16, tf_lite_model_32, onnx_model, image_paths)
    
#     print("\n=== 벤치마크 결과 ===")
#     # 성능 측정 출력 (시간)
#     if tf_lite_16_times:
#         print(f"TensorFlow Lite Float16 평균 추론 시간: {mean(tf_lite_16_times[10:]) * 1000:.2f}ms")
#         print(f"TensorFlow Lite Float16 표준 편차: {stdev(tf_lite_16_times[10:]) * 1000:.2f}ms")
#     if tf_lite_32_times:
#         print(f"TensorFlow Lite Float32 평균 추론 시간: {mean(tf_lite_32_times[10:]) * 1000:.2f}ms")
#         print(f"TensorFlow Lite Float32 표준 편차: {stdev(tf_lite_32_times[10:]) * 1000:.2f}ms")
#     if onnx_times:
#         print(f"ONNX 평균 추론 시간: {mean(onnx_times[10:]) * 1000:.2f}ms")
#         print(f"ONNX 표준 편차: {stdev(onnx_times[10:]) * 1000:.2f}ms")
    
#     # 메모리 사용 출력
#     if tf_lite_16_memories:
#         print(f"TensorFlow Lite Float16 평균 메모리 사용량: {mean(tf_lite_16_memories[10:]) / 1024:.2f} KB")
#     if tf_lite_32_memories:
#         print(f"TensorFlow Lite Float32 평균 메모리 사용량: {mean(tf_lite_32_memories[10:]) / 1024:.2f} KB")
#     if onnx_memories:
#         print(f"ONNX 평균 메모리 사용량: {mean(onnx_memories[10:]) / 1024:.2f} KB")
    
#     # 벤치마크 그래프 저장
#     save_benchmark_graphs(tf_lite_16_times, tf_lite_32_times, onnx_times, 
#                           tf_lite_16_memories, tf_lite_32_memories, onnx_memories)

# if __name__ == "__main__":
#     main()


import time
import os
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import psutil
from statistics import mean, stdev
import matplotlib.pyplot as plt

# 모델 경로 설정
save_dir = Path("/home/jetson/Downloads/yolov/yolov8n/yolov8n_saved_model/")
tflite_float16_path = save_dir / "yolov8n_float16.tflite"
tflite_float32_path = save_dir / "yolov8n_float32.tflite"
onnx_model_path = "yolov8n.onnx"
pt_model_path = "yolov8n.pt"

# 데이터셋 경로 설정
dataset_path = "coco128"

def load_models():
    """YOLOv8, TFLite, ONNX 모델을 로드합니다."""
    # YOLOv8 PyTorch 모델
    yolo_model = YOLO(pt_model_path)
    yolo_model.to('cpu')  # CPU로 이동

    # TFLite 모델 로드
    tf_lite_interpreter_16 = tf.lite.Interpreter(model_path=str(tflite_float16_path))
    tf_lite_interpreter_16.allocate_tensors()
    
    tf_lite_interpreter_32 = tf.lite.Interpreter(model_path=str(tflite_float32_path))
    tf_lite_interpreter_32.allocate_tensors()

    ort_sess = ort.InferenceSession(str(onnx_model_path))  # providers 매개변수 제거

    # # CUDA를 우선적으로 사용
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']


    # # ONNX 모델 로드
    # ort_sess = ort.InferenceSession(
    # str(onnx_model_path),
    # providers=providers
    # )

    return yolo_model, tf_lite_interpreter_16, tf_lite_interpreter_32, ort_sess

def prepare_image(image_path, input_size=(640, 640)):
    """이미지를 모델 입력 형식에 맞게 준비."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img = np.array(img, dtype=np.float32) / 255.0  # 정규화
    return img

def prepare_image_tflite(image, input_details):
    """TensorFlow Lite 입력 데이터 변환."""
    img = np.expand_dims(image, axis=0)
    return img.astype(np.float32)

def benchmark_inference(models, image_paths, num_iterations=1):
    """모델별 추론 속도와 메모리 사용량을 측정합니다."""
    times, memories = {}, {}
    process = psutil.Process(os.getpid())

    for model_name, model in models.items():
        model_times, model_memories = [], []
        print(f"Running {model_name} inference...")
        
        for image_path in tqdm(image_paths, desc=model_name):
            img = prepare_image(image_path)
            
            if "TFLite" in model_name:
                interpreter = model
                input_details = interpreter.get_input_details()
                interpreter.set_tensor(input_details[0]['index'], prepare_image_tflite(img, input_details))
                
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    interpreter.invoke()
                    model_times.append(time.perf_counter() - start_time)
                    model_memories.append(process.memory_info().rss)
            
            elif model_name == "ONNX":
                input_name = model.get_inputs()[0].name
                img_onnx = np.transpose(img, (2, 0, 1))
                img_onnx = np.expand_dims(img_onnx, axis=0)

                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    model.run(None, {input_name: img_onnx})
                    model_times.append(time.perf_counter() - start_time)
                    model_memories.append(process.memory_info().rss)

            elif model_name == "pytorch":
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    model.predict(img, save=False, verbose=False, cache=False)
                    model_times.append(time.perf_counter() - start_time)
                    model_memories.append(process.memory_info().rss)

        times[model_name] = model_times
        memories[model_name] = model_memories

    return times, memories

# def calculate_map(models, dataset_path):
#     """각 모델에 대한 mAP 계산."""
#     maps = {}
#     for model_name, model in models.items():
#         if model_name == "YOLOv8 (.pt)":
#             print(f"Calculating mAP for {model_name}...")
#             results = model.val(data=dataset_path)
#             maps[model_name] = (results.box.map50, results.box.map)
#         else:
#             # TFLite 및 ONNX의 mAP 측정을 위해서는 맞춤 구현 필요
#             # 추론 결과를 기반으로 직접 계산하는 로직 추가
#             print(f"mAP calculation not directly supported for {model_name}.")
#             maps[model_name] = (None, None)

#     return maps

def save_benchmark_results(times, memories):
    """추론 속도, 메모리 사용량 및 mAP 점수 저장."""
    model_names = list(times.keys())
    avg_times = [mean(times[model_name]) * 1000 for model_name in model_names]
    avg_memories = [mean(memories[model_name]) / 1024 for model_name in model_names]
    # map_scores = [maps[model_name][0] if maps[model_name][0] else 0 for model_name in model_names]

    # 추론 시간 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, avg_times, color=['blue', 'orange', 'green', 'red'])
    plt.title("Average Inference Time (ms)")
    plt.ylabel("Time (ms)")
    plt.savefig("inference_times.png")
    plt.close()

    # 메모리 사용량 그래프
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, avg_memories, color=['blue', 'orange', 'green', 'red'])
    plt.title("Average Memory Usage (KB)")
    plt.ylabel("Memory (KB)")
    plt.savefig("memory_usage.png")
    plt.close()

    # # mAP 점수 그래프
    # plt.figure(figsize=(10, 6))
    # plt.bar(model_names, map_scores, color=['blue', 'orange', 'green', 'red'])
    # plt.title("mAP50 Scores")
    # plt.ylabel("mAP50")
    # plt.savefig("map_scores.png")
    # plt.close()

def main():
    print("모델 로드 중...")
    yolo_model, tf_lite_16, tf_lite_32, onnx_model = load_models()

    models = {
        "pytorch": yolo_model,
        "TFLite Float16": tf_lite_16,
        "TFLite Float32": tf_lite_32,
        "ONNX": onnx_model,
    }

    print("이미지 파일 로드 중...")
    image_folder = Path(dataset_path) / "images/train2017"
    image_paths = [str(p) for p in image_folder.glob("*.jpg")]

    print("추론 벤치마크 시작...")
    times, memories = benchmark_inference(models, image_paths)

    # print("mAP 계산 중...")
    # maps = calculate_map(models, dataset_path)

    print("\n=== 결과 ===")
    for model_name in models.keys():
        print(f"{model_name}:")
        print(f"  - 평균 추론 시간: {mean(times[model_name]) * 1000:.2f} ms")
        print(f"  - 평균 메모리 사용량: {mean(memories[model_name]) / 1024:.2f} KB")
        # if maps[model_name][0]:
        #     print(f"  - mAP50: {maps[model_name][0]:.4f}, mAP50-95: {maps[model_name][1]:.4f}")
        # else:
        #     print(f"  - mAP: 지원되지 않음")

    save_benchmark_results(times, memories)

if __name__ == "__main__":
    main()
