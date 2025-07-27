import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.detection_objects import detection_objects_config


class DetectionService:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.mp_objectron = mp.solutions.objectron
        self.config = detection_objects_config
        
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
        
        if detection_object.model_type == 'hands':
            return self.detect_hands(image, confidence_threshold)
        elif detection_object.model_type == 'face_detection':
            return self.detect_faces(image, confidence_threshold)
        elif detection_object.model_type == 'pose':
            return self.detect_pose(image, confidence_threshold)
        elif detection_object.model_type == 'objectron':
            return self.detect_objects(image, object_type, confidence_threshold)
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {detection_object.model_type}") 