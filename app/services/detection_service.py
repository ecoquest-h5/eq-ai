import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
import requests
import sys
import os
from ultralytics import YOLO
from PIL import Image

# PyTorch와 transformers를 조건부로 import
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers 라이브러리 로드 실패: {e}")
    TRANSFORMERS_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.detection_objects import detection_objects_config


class DetectionService:
    def __init__(self, yolo_model_type: str = "s"):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.mp_objectron = mp.solutions.objectron
        self.config = detection_objects_config
        
        # YOLO v8 모델 초기화
        self.yolo_model_type = yolo_model_type
        self.yolo_model = None
        self._initialize_yolo_model()
        
        # Transformer 모델 초기화
        self.transformer_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            self._initialize_transformer_model()
        else:
            print("Transformers 라이브러리를 사용할 수 없어 Transformer 모델을 건너뜁니다.")
        
    def _initialize_yolo_model(self):
        """YOLO v8 모델을 초기화합니다."""
        try:
            model_name = f"yolov8{self.yolo_model_type}"
            self.yolo_model = YOLO(model_name)
            print(f"YOLO 모델 로드 완료: {model_name}")
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {e}")
            # 기본 모델로 fallback
            try:
                self.yolo_model = YOLO("yolov8s")
                print("기본 YOLO 모델로 fallback: yolov8s")
            except Exception as fallback_error:
                print(f"YOLO 모델 fallback 실패: {fallback_error}")
                self.yolo_model = None
    
    def _initialize_transformer_model(self):
        """Transformer 모델을 초기화합니다."""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers 라이브러리를 사용할 수 없습니다.")
            return
            
        try:
            # torch 모듈을 다시 import (스코프 문제 해결)
            import torch
            
            # GPU 사용 가능 여부 확인
            device = 0 if torch.cuda.is_available() else -1
            print(f"Transformer 모델 디바이스: {'GPU' if device == 0 else 'CPU'}")
            
            self.transformer_pipeline = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=device
            )
            print("Transformer 모델 로드 완료: facebook/detr-resnet-50")
        except Exception as e:
            print(f"Transformer 모델 로드 실패: {e}")
            self.transformer_pipeline = None
        
    def download_image_from_url(self, url: str) -> bytes:
        """URL에서 이미지를 다운로드합니다."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            raise Exception(f"이미지 다운로드 실패: {str(e)}")
    
    def process_image_from_url(self, url: str) -> np.ndarray:
        """URL에서 이미지를 다운로드하고 OpenCV 형식으로 변환합니다."""
        image_bytes = self.download_image_from_url(url)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise Exception("이미지를 읽을 수 없습니다")
        
        return image
    
    def detect_with_transformer(self, image: np.ndarray, target_class_id: int, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        """Transformer 모델을 사용하여 객체를 감지합니다."""
        if not TRANSFORMERS_AVAILABLE:
            raise Exception("Transformers 라이브러리를 사용할 수 없습니다.")
            
        if self.transformer_pipeline is None:
            raise Exception("Transformer 모델이 초기화되지 않았습니다")
        
        try:
            # OpenCV 이미지를 PIL 이미지로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Transformer 모델로 예측
            results = self.transformer_pipeline(pil_image)
            
            # COCO 클래스 이름을 ID로 변환하는 매핑
            coco_name_to_id = {
                'handbag': 31,
                'tv': 72,
                'tie': 27,
                'backpack': 27,  # clothes 대신 backpack 사용
                'suitcase': 31,  # handbag 대안
                'remote': 72,    # tv 대안
                'person': 1,     # clothes 감지용
                'umbrella': 28,  # clothes 감지용
                # trash_bin 감지를 위한 추가 대안들 (분리수거, 쓰레기 버리기 관련)
                'monitor': 72,   # tv/monitor
                'screen': 72,    # tv/screen
                'computer': 72,  # tv/computer
                'laptop': 72,    # tv/laptop
                'box': 72,       # tv/box (쓰레기통 모양)
                'container': 72, # tv/container
                'sink': 72,      # 싱크대 (분리수거함과 유사)
                'refrigerator': 73, # 냉장고 (큰 용기)
                'microwave': 69, # 전자레인지 (용기 모양)
                'oven': 70,      # 오븐 (용기 모양)
                'toaster': 71,   # 토스터 (용기 모양)
                'vase': 76,      # 화분 (용기 모양)
                'bowl': 46,      # 그릇 (용기 모양)
                'cup': 42,       # 컵 (용기 모양)
                'bottle': 40,    # 병 (재활용 관련)
                'wine glass': 41, # 와인잔 (용기 모양)
                'book': 74,      # 책 (재활용 관련)
                'cell phone': 68, # 휴대폰 (재활용 관련)
                'keyboard': 67,  # 키보드 (재활용 관련)
                'mouse': 65,     # 마우스 (재활용 관련)
                'remote': 66,    # 리모컨 (재활용 관련)
                'clock': 75,     # 시계 (재활용 관련)
                'scissors': 77,  # 가위 (재활용 관련)
                'toothbrush': 80, # 칫솔 (재활용 관련)
                'hair drier': 79, # 헤어드라이어 (재활용 관련)
            }
            
            # 디버깅: 감지된 모든 객체 출력
            if results:
                print(f"감지된 객체들: {[result.get('label', 'unknown') for result in results]}")
            
            # clothes 감지를 위한 특별 처리
            if target_class_id == 27:  # clothes 클래스
                for result in results:
                    label_name = result.get('label', '').lower()
                    score = result.get('score', 0.0)
                    
                    # person 또는 umbrella가 감지되면 clothes로 인식
                    if label_name in ['person', 'umbrella'] and score >= confidence_threshold:
                        bbox = result['box']
                        x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                        bounding_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        landmarks = [
                            [float(x1), float(y1)],  # 좌상단
                            [float(x2), float(y1)],  # 우상단
                            [float(x2), float(y2)],  # 우하단
                            [float(x1), float(y2)]   # 좌하단
                        ]
                        
                        return True, score, bounding_box, landmarks
                
                return False, 0.0, None, None
            
            # trash_bin 감지를 위한 특별 처리
            if target_class_id == 72:  # trash_bin 클래스
                for result in results:
                    label_name = result.get('label', '').lower()
                    score = result.get('score', 0.0)
                    
                    # 분리수거, 쓰레기 버리기 관련 클래스들이 감지되면 trash_bin으로 인식
                    trash_related_classes = [
                        # 전자기기/용기류 (쓰레기통 모양)
                        'tv', 'monitor', 'screen', 'computer', 'laptop', 'box', 'container',
                        'sink', 'refrigerator', 'microwave', 'oven', 'toaster', 'vase',
                        # 용기류
                        'bowl', 'cup', 'bottle', 'wine glass',
                        # 재활용 관련 물품들
                        'book', 'cell phone', 'keyboard', 'mouse', 'remote', 'clock',
                        'scissors', 'toothbrush', 'hair drier'
                    ]
                    
                    if label_name in trash_related_classes and score >= confidence_threshold:
                        bbox = result['box']
                        x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                        bounding_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        landmarks = [
                            [float(x1), float(y1)],  # 좌상단
                            [float(x2), float(y1)],  # 우상단
                            [float(x2), float(y2)],  # 우하단
                            [float(x1), float(y2)]   # 좌하단
                        ]
                        
                        print(f"trash_bin 감지됨: {label_name} (신뢰도: {score}) - 분리수거/재활용 관련")
                        return True, score, bounding_box, landmarks
                
                return False, 0.0, None, None
            
            # bottle 감지를 위한 특별 처리
            if target_class_id == 40:  # bottle 클래스
                for result in results:
                    label_name = result.get('label', '').lower()
                    score = result.get('score', 0.0)
                    
                    # bottle, cup, vase가 감지되면 bottle로 인식
                    bottle_related_classes = ['bottle', 'cup', 'vase']
                    
                    if label_name in bottle_related_classes and score >= confidence_threshold:
                        bbox = result['box']
                        x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                        bounding_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        landmarks = [
                            [float(x1), float(y1)],  # 좌상단
                            [float(x2), float(y1)],  # 우상단
                            [float(x2), float(y2)],  # 우하단
                            [float(x1), float(y2)]   # 좌하단
                        ]
                        
                        print(f"bottle 감지됨: {label_name} (신뢰도: {score}) - bottle/cup/vase 관련")
                        return True, score, bounding_box, landmarks
                
                return False, 0.0, None, None
            
            # 일반적인 감지 처리 (handbag)
            for result in results:
                # label 키에서 클래스 이름 가져오기
                label_name = result.get('label', '').lower()
                score = result.get('score', 0.0)
                
                # 클래스 이름을 ID로 변환
                detected_class_id = coco_name_to_id.get(label_name)
                
                if detected_class_id == target_class_id and score >= confidence_threshold:
                    # 바운딩 박스 좌표
                    bbox = result['box']
                    x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                    bounding_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    
                    # 랜드마크 (바운딩 박스의 4개 모서리)
                    landmarks = [
                        [float(x1), float(y1)],  # 좌상단
                        [float(x2), float(y1)],  # 우상단
                        [float(x2), float(y2)],  # 우하단
                        [float(x1), float(y2)]   # 좌하단
                    ]
                    
                    return True, score, bounding_box, landmarks
            
            return False, 0.0, None, None
            
        except Exception as e:
            print(f"Transformer 감지 중 오류 발생: {e}")
            return False, 0.0, None, None
    
    def detect_with_yolo(self, image: np.ndarray, target_class: str, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        """YOLO v8을 사용하여 객체를 감지합니다."""
        if self.yolo_model is None:
            raise Exception("YOLO 모델이 초기화되지 않았습니다")
        
        try:
            # YOLO 모델로 예측
            results = self.yolo_model(image, verbose=False)
            
            # COCO 클래스 이름 매핑
            coco_classes = self.yolo_model.names
            
            # 타겟 클래스 찾기
            target_class_id = None
            for class_id, class_name in coco_classes.items():
                if class_name.lower() == target_class.lower():
                    target_class_id = class_id
                    break
            
            if target_class_id is None:
                return False, 0.0, None, None
            
            # 결과에서 타겟 클래스 찾기
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == target_class_id and confidence >= confidence_threshold:
                            # 바운딩 박스 좌표
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bounding_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            
                            # 랜드마크 (바운딩 박스의 4개 모서리)
                            landmarks = [
                                [float(x1), float(y1)],  # 좌상단
                                [float(x2), float(y1)],  # 우상단
                                [float(x2), float(y2)],  # 우하단
                                [float(x1), float(y2)]   # 좌하단
                            ]
                            
                            return True, confidence, bounding_box, landmarks
            
            return False, 0.0, None, None
            
        except Exception as e:
            print(f"YOLO 감지 중 오류 발생: {e}")
            return False, 0.0, None, None
        
    def detect_hands(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold
        ) as hands:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                
                return True, 0.8, None, landmarks
            
            return False, 0.0, None, None
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        with self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence_threshold
        ) as face_detection:
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.detections:
                detection = results.detections[0]
                confidence = detection.score[0]
                bbox = detection.location_data.relative_bounding_box
                bounding_box = [bbox.xmin, bbox.ymin, bbox.width, bbox.height]
                
                landmarks = []
                for landmark in detection.location_data.relative_keypoints:
                    landmarks.append([landmark.x, landmark.y])
                
                return True, confidence, bounding_box, landmarks
            
            return False, 0.0, None, None
    
    def detect_pose(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=confidence_threshold
        ) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                return True, 0.8, None, landmarks
            
            return False, 0.0, None, None
    
    def detect_objects(self, image: np.ndarray, object_type: str, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        detection_object = self.config.get_object(object_type)
        
        if detection_object.model_type != 'objectron':
            raise ValueError(f"잘못된 모델 타입입니다: {detection_object.model_type}")
        
        with self.mp_objectron.Objectron(
            static_image_mode=True, 
            max_num_objects=5, 
            min_detection_confidence=confidence_threshold, 
            model_name=detection_object.model_name
        ) as objectron:
            results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.detected_objects:
                detected_object = results.detected_objects[0]
                
                confidence = 0.8
                try:
                    if hasattr(detected_object, 'score') and detected_object.score:
                        confidence = detected_object.score[0]
                except (AttributeError, IndexError):
                    pass
                
                landmarks = []
                try:
                    if hasattr(detected_object, 'landmarks_2d') and detected_object.landmarks_2d:
                        for landmark in detected_object.landmarks_2d.landmark:
                            landmarks.append([landmark.x, landmark.y])
                except (AttributeError, IndexError):
                    pass
                
                return True, confidence, None, landmarks
            
            return False, 0.0, None, None
    
    def detect_object_from_url(self, url: str, object_type: str, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        """URL에서 이미지를 다운로드하고 물체를 감지합니다."""
        image = self.process_image_from_url(url)
        detection_object = self.config.get_object(object_type)
        
        if detection_object.model_type == 'transformer':
            if not TRANSFORMERS_AVAILABLE:
                raise Exception("Transformers 라이브러리를 사용할 수 없어 Transformer 기반 객체를 감지할 수 없습니다.")
            return self.detect_with_transformer(image, detection_object.coco_class_id, confidence_threshold)
        elif detection_object.model_type == 'yolo':
            return self.detect_with_yolo(image, detection_object.model_name, confidence_threshold)
        elif detection_object.model_type == 'hands':
            return self.detect_hands(image, confidence_threshold)
        elif detection_object.model_type == 'face_detection':
            return self.detect_faces(image, confidence_threshold)
        elif detection_object.model_type == 'pose':
            return self.detect_pose(image, confidence_threshold)
        elif detection_object.model_type == 'objectron':
            return self.detect_objects(image, object_type, confidence_threshold)
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {detection_object.model_type}") 