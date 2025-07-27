import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import io


class DetectionService:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.mp_objectron = mp.solutions.objectron
        
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
                    hand_points = []
                    for landmark in hand_landmarks.landmark:
                        hand_points.append([landmark.x, landmark.y, landmark.z])
                    landmarks.extend(hand_points)
                
                confidence = 0.8
                return True, confidence, None, landmarks
            
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
                
                confidence = 0.8
                return True, confidence, None, landmarks
            
            return False, 0.0, None, None
    
    def detect_objects(self, image: np.ndarray, object_type: str, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        objectron_models = {
            'chair': self.mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=confidence_threshold, model_name='Chair'),
            'cup': self.mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=confidence_threshold, model_name='Cup'),
            'camera': self.mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=confidence_threshold, model_name='Camera'),
            'shoe': self.mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=confidence_threshold, model_name='Shoe')
        }
        
        if object_type not in objectron_models:
            return False, 0.0, None, None
        
        with objectron_models[object_type] as objectron:
            results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.detected_objects:
                detected_object = results.detected_objects[0]
                confidence = detected_object.score[0]
                
                landmarks = []
                for landmark in detected_object.landmarks_2d.landmark:
                    landmarks.append([landmark.x, landmark.y])
                
                return True, confidence, None, landmarks
            
            return False, 0.0, None, None
    
    def detect_object(self, image_bytes: bytes, object_type: str, confidence_threshold: float = 0.5) -> Tuple[bool, float, Optional[List[float]], Optional[List[List[float]]]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if object_type == 'hand':
            return self.detect_hands(image, confidence_threshold)
        elif object_type == 'face':
            return self.detect_faces(image, confidence_threshold)
        elif object_type == 'pose':
            return self.detect_pose(image, confidence_threshold)
        else:
            return self.detect_objects(image, object_type, confidence_threshold) 