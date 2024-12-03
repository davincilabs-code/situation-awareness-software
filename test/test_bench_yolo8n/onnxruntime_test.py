# import os
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# from PIL import Image, ImageDraw
# import onnxruntime as ort

# # 모델 경로 설정
# onnx_model_path = "yolov8n.onnx"

# # 데이터셋 경로 설정
# dataset_path = "coco128"
# results_folder = Path("results")
# results_folder.mkdir(exist_ok=True)

# def load_onnx_model():
#     """ONNX 모델 로드"""
#     ort_sess = ort.InferenceSession(str(onnx_model_path))
#     return ort_sess

# def prepare_image(image_path, input_size=(640, 640)):
#     """이미지를 모델 입력 형식에 맞게 준비"""
#     img = Image.open(image_path).convert('RGB')
#     original_size = img.size  # 원본 크기 저장
#     img = img.resize(input_size)
#     img = np.array(img, dtype=np.float32) / 255.0  # 정규화
#     img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
#     img = np.expand_dims(img, axis=0)  # 배치 차원 추가
#     return img, original_size

# def run_onnx_inference(onnx_model, image_paths):
#     """ONNX 모델 추론 및 결과 저장"""
#     input_name = onnx_model.get_inputs()[0].name
#     output_name = onnx_model.get_outputs()[0].name
#     input_size = (640, 640)  # 모델 입력 크기
    
#     for image_path in tqdm(image_paths, desc="ONNX Inference"):
#         img, original_size = prepare_image(image_path, input_size)
        
#         # 추론
#         outputs = onnx_model.run([output_name], {input_name: img})
        
#         # 출력 후처리 및 결과 저장
#         save_results(image_path, outputs[0], original_size)

# def save_results(image_path, outputs, original_size):
#     """
#     YOLOv8 ONNX 모델 출력 처리 및 바운딩 박스 표시.
#     """
#     # 출력 텐서 구조: (1, 84, 8400)
#     predictions = np.squeeze(outputs)  # 배치 차원 제거, (84, 8400)

#     # 바운딩 박스 좌표 분리 (0~3: x_center, y_center, width, height)
#     box_predictions = predictions[:4, :].T  # (8400, 4)

#     # 클래스 확률 분리 (4~83: class probabilities)
#     scores = predictions[4:, :].T  # (8400, 80)

#     # 최고 클래스와 확률 계산
#     class_ids = np.argmax(scores, axis=1)  # 각 anchor box의 최고 클래스 인덱스 (8400,)
#     confidences = np.max(scores, axis=1)  # 각 anchor box의 최고 클래스 확률 (8400,)

#     # Confidence threshold 적용
#     valid_indices = confidences > 0.5  # 50% 이상의 확신만 사용
#     box_predictions = box_predictions[valid_indices]
#     confidences = confidences[valid_indices]
#     class_ids = class_ids[valid_indices]

#     # 좌표 변환: 정규화된 값 -> 원본 이미지 크기
#     box_predictions[:, 0] = (box_predictions[:, 0] - box_predictions[:, 2] / 2) * original_size[0]  # x1
#     box_predictions[:, 1] = (box_predictions[:, 1] - box_predictions[:, 3] / 2) * original_size[1]  # y1
#     box_predictions[:, 2] = (box_predictions[:, 0] + box_predictions[:, 2]) * original_size[0]  # x2
#     box_predictions[:, 3] = (box_predictions[:, 1] + box_predictions[:, 3]) * original_size[1]  # y2

#     # NMS 적용 (중복 박스 제거)
#     nms_indices = nms(box_predictions, confidences, iou_threshold=0.4)
#     box_predictions = box_predictions[nms_indices]
#     confidences = confidences[nms_indices]
#     class_ids = class_ids[nms_indices]

#     # 원본 이미지 로드
#     img = Image.open(image_path).convert("RGB")
#     draw = ImageDraw.Draw(img)

#     # 결과 처리 및 표시
#     for box, conf, class_id in zip(box_predictions, confidences, class_ids):
#         x1, y1, x2, y2 = map(int, box)
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
#         draw.text((x1, y1), f"Class {class_id}: {conf:.2f}", fill="red")

#     # 결과 이미지 저장
#     result_path = results_folder / Path(image_path).name
#     img.save(result_path)


# def nms(boxes, scores, iou_threshold=0.5):
#     """Non-Maximum Suppression (NMS) 구현.
    
#     Args:
#         boxes (np.ndarray): 바운딩 박스 좌표 (N, 4).
#         scores (np.ndarray): 각 박스의 점수 (N,).
#         iou_threshold (float): IOU 임계값.

#     Returns:
#         np.ndarray: 선택된 박스의 인덱스.
#     """
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]

#     areas = (x2 - x1) * (y2 - y1)  # 각 박스의 면적
#     order = scores.argsort()[::-1]  # 점수를 기준으로 정렬 (내림차순)

#     keep = []  # 유지할 박스 인덱스
#     while order.size > 0:
#         i = order[0]  # 가장 높은 점수의 인덱스
#         keep.append(i)

#         # IOU 계산
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h
#         iou = inter / (areas[i] + areas[order[1:]] - inter)

#         # IOU 임계값보다 작은 박스만 남김
#         inds = np.where(iou <= iou_threshold)[0]
#         order = order[inds + 1]

#     return keep

# def main():
#     print("ONNX 모델 로드 중...")
#     onnx_model = load_onnx_model()

#     print("이미지 파일 로드 중...")
#     image_folder = Path(dataset_path) / "images/train2017"
#     image_paths = [str(p) for p in image_folder.glob("*.jpg")]

#     print("ONNX 모델 추론 시작...")

#     # 디버깅: 이미지 경로와 모델 로드 확인
#     print(f"총 이미지 개수: {len(image_paths)}")
#     if not image_paths:
#         print("이미지 파일을 찾을 수 없습니다.")
#         return

#     run_onnx_inference(onnx_model, image_paths)

#     print("추론 완료. 결과가 results 폴더에 저장되었습니다.")

# if __name__ == "__main__":
#     main()

import os
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from tqdm import tqdm

# 설정
dataset_path = "coco128"
image_folder = Path(dataset_path) / "images/train2017"
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

# ONNX 모델 로드
onnx_model_path = "yolov8n.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

def compute_iou(box1, box2):
    """Compute IoU (Intersection over Union) between two boxes.
    
    Args:
        box1: Array [x1, y1, x2, y2]
        box2: Array [x1, y1, x2, y2]
        
    Returns:
        IoU: Float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union = box1_area + box2_area - intersection

    # Avoid division by zero
    if union == 0:
        return 0

    return intersection / union

def nms_with_highest_confidence(boxes, scores, iou_threshold):
    """Perform Non-Maximum Suppression (NMS) based on highest confidence.
    
    Args:
        boxes: Array of shape (N, 4) containing [x1, y1, x2, y2] for each box.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: Float, IoU threshold for NMS.
        
    Returns:
        selected_indices: List of indices of boxes to keep.
    """
    indices = np.argsort(scores)[::-1]  # Sort by confidence scores in descending order
    selected_indices = []

    while len(indices) > 0:
        current_index = indices[0]
        selected_indices.append(current_index)
        if len(indices) == 1:
            break

        remaining_indices = indices[1:]
        current_box = boxes[current_index]

        ious = np.array([compute_iou(current_box, boxes[i]) for i in remaining_indices])

        # Keep boxes with IoU less than the threshold
        indices = remaining_indices[ious < iou_threshold]

    return selected_indices

# 이미지 추론 및 바운딩 박스 그리기
def infer_and_draw_bboxes(image_path):
    # 이미지 불러오기
    image = cv2.imread(str(image_path))
    h, w, _ = image.shape

    # 이미지 전처리
    img_resized = cv2.resize(image, (640, 640))
    img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # ONNX 모델 추론
    ort_inputs = {ort_session.get_inputs()[0].name: img_input}
    outputs = ort_session.run(None, ort_inputs)

    # 출력 처리
    predictions = outputs[0]  # Shape: (1, 84, 8400)

    # Transpose the output to shape (1, 8400, 84)
    predictions = np.transpose(predictions, (0, 2, 1))
    predictions = predictions[0]  # Remove batch dimension -> Shape: (8400, 84)

    # Extract boxes, scores, and class probabilities
    boxes = predictions[:, :4]  # (x_center, y_center, width, height)
    scores = predictions[:, 4]  # Objectness confidence
    class_probs = predictions[:, 5:]  # Class probabilities

    # Apply confidence threshold
    confidence_threshold = 0.3
    indices = np.where(scores > confidence_threshold)[0]
    boxes = boxes[indices]
    scores = scores[indices]
    class_probs = class_probs[indices]
    class_ids = np.argmax(class_probs, axis=1)

    # Rescale boxes to original image dimensions
    boxes[:, 0] *= w  # x_center
    boxes[:, 1] *= h  # y_center
    boxes[:, 2] *= w  # width
    boxes[:, 3] *= h  # height

    # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Clip boxes to image dimensions
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h - 1)

    # NMS 적용
    nms_indices = nms_with_highest_confidence(boxes_xyxy, scores, iou_threshold=0.5)
    boxes_xyxy = boxes_xyxy[nms_indices]
    scores = scores[nms_indices]
    class_ids = class_ids[nms_indices]

    # 바운딩 박스 그리기
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        confidence = scores[i]
        class_id = class_ids[i]
        print(f"Detected box: {x1, y1, x2, y2}, Confidence: {confidence:.2f}, Class ID: {class_id}")

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put class label
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 저장
    result_path = results_folder / image_path
    cv2.imwrite(str(result_path), image)

    
def compute_iou(box1, box2):
    """Compute IoU (Intersection over Union) between two boxes.
    
    Args:
        box1: Array [x1, y1, x2, y2]
        box2: Array [x1, y1, x2, y2]
        
    Returns:
        IoU: Float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union = box1_area + box2_area - intersection

    # Avoid division by zero
    if union == 0:
        return 0

    return intersection / union


def nms_with_highest_confidence(boxes, scores, iou_threshold):
    """Perform Non-Maximum Suppression (NMS) based on highest confidence.
    
    Args:
        boxes: Array of shape (N, 4) containing [x1, y1, x2, y2] for each box.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: Float, IoU threshold for NMS.
        
    Returns:
        selected_indices: List of indices of boxes to keep.
    """
    indices = np.argsort(scores)[::-1]  # Sort by confidence scores in descending order
    selected_indices = []

    while len(indices) > 0:
        current_index = indices[0]
        selected_indices.append(current_index)
        if len(indices) == 1:
            break

        remaining_indices = indices[1:]
        current_box = boxes[current_index]

        ious = np.array([compute_iou(current_box, boxes[i]) for i in remaining_indices])

        # Keep boxes with IoU less than the threshold
        indices = remaining_indices[ious < iou_threshold]

    return selected_indices


# 이미지 추론 및 바운딩 박스 그리기
def infer_and_draw_bboxes(image_path):
    # 이미지 불러오기
    image = cv2.imread(str(image_path))
    h, w, _ = image.shape

    # 이미지 전처리
    img_resized = cv2.resize(image, (640, 640))
    img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # ONNX 모델 추론
    ort_inputs = {ort_session.get_inputs()[0].name: img_input}
    outputs = ort_session.run(None, ort_inputs)

    # 출력 처리
    predictions = outputs[0]  # Shape: (1, 84, 8400)

    # Transpose the output to shape (1, 8400, 84)
    predictions = np.transpose(predictions, (0, 2, 1))
    predictions = predictions[0]  # Remove batch dimension -> Shape: (8400, 84)

    # Extract boxes, scores, and class probabilities
    boxes = predictions[:, :4]  # (x_center, y_center, width, height)
    scores = predictions[:, 4]  # Objectness confidence
    class_probs = predictions[:, 5:]  # Class probabilities

    # Apply confidence threshold
    confidence_threshold = 0.3
    indices = np.where(scores > confidence_threshold)[0]
    boxes = boxes[indices]
    scores = scores[indices]
    class_probs = class_probs[indices]
    class_ids = np.argmax(class_probs, axis=1)

    # Rescale boxes to original image dimensions
    boxes[:, 0] *= w  # x_center
    boxes[:, 1] *= h  # y_center
    boxes[:, 2] *= w  # width
    boxes[:, 3] *= h  # height

    # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Clip boxes to image dimensions
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h - 1)

    # NMS 적용
    nms_indices = nms_with_highest_confidence(boxes_xyxy, scores, iou_threshold=0.5)
    boxes_xyxy = boxes_xyxy[nms_indices]
    scores = scores[nms_indices]
    class_ids = class_ids[nms_indices]

    # 바운딩 박스 그리기
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        confidence = scores[i]
        class_id = class_ids[i]
        print(f"Detected box: {x1, y1, x2, y2}, Confidence: {confidence:.2f}, Class ID: {class_id}")

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put class label
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 저장
    result_path = results_folder / image_path.name
    cv2.imwrite(str(result_path), image)

# 메인 함수
def main():
    # 이미지 폴더 내 모든 이미지 파일 목록 가져오기
    image_files = list(image_folder.glob("*.jpg"))  # 필요에 따라 확장자 변경

    # tqdm을 사용하여 진행 상황을 표시
    for image_file in tqdm(image_files, desc="Processing Images"):
        infer_and_draw_bboxes(image_file)

# 스크립트 실행 시 메인 함수 호출
if __name__ == "__main__":
    main()
