from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DetectionObject:
    name: str
    model_type: str
    model_name: str
    description: str
    supported: bool = True


class DetectionObjectsConfig:
    """감지 가능한 물체들의 설정을 관리합니다."""
    
    def __init__(self):
        self._objects = self._initialize_objects()
    
    def _initialize_objects(self) -> Dict[str, DetectionObject]:
        """감지 가능한 물체들을 초기화합니다."""
        return {
            'hand': DetectionObject(
                name='hand',
                model_type='hands',
                model_name='Hands',
                description='손 감지 (MediaPipe Hands)'
            ),
            'face': DetectionObject(
                name='face',
                model_type='face_detection',
                model_name='FaceDetection',
                description='얼굴 감지 (MediaPipe Face Detection)'
            ),
            'pose': DetectionObject(
                name='pose',
                model_type='pose',
                model_name='Pose',
                description='자세 감지 (MediaPipe Pose)'
            ),
            'chair': DetectionObject(
                name='chair',
                model_type='objectron',
                model_name='Chair',
                description='의자 감지 (MediaPipe Objectron)'
            ),
            'cup': DetectionObject(
                name='cup',
                model_type='objectron',
                model_name='Cup',
                description='컵 감지 (MediaPipe Objectron)'
            ),
            'camera': DetectionObject(
                name='camera',
                model_type='objectron',
                model_name='Camera',
                description='카메라 감지 (MediaPipe Objectron)'
            ),
            'shoe': DetectionObject(
                name='shoe',
                model_type='objectron',
                model_name='Shoe',
                description='신발 감지 (MediaPipe Objectron)'
            ),
            'bottle': DetectionObject(
                name='bottle',
                model_type='objectron',
                model_name='Bottle',
                description='병 감지 (MediaPipe Objectron)'
            ),
            'bowl': DetectionObject(
                name='bowl',
                model_type='objectron',
                model_name='Bowl',
                description='그릇 감지 (MediaPipe Objectron)'
            )
        }
    
    def get_object(self, object_name: str) -> DetectionObject:
        """물체 이름으로 DetectionObject를 가져옵니다."""
        if object_name not in self._objects:
            raise ValueError(f"지원하지 않는 물체 타입입니다: {object_name}")
        return self._objects[object_name]
    
    def get_all_objects(self) -> Dict[str, DetectionObject]:
        """모든 감지 가능한 물체들을 반환합니다."""
        return self._objects.copy()
    
    def get_supported_objects(self) -> List[str]:
        """지원되는 물체 이름들의 리스트를 반환합니다."""
        return [obj.name for obj in self._objects.values() if obj.supported]
    
    def get_objects_by_type(self, model_type: str) -> List[DetectionObject]:
        """모델 타입별로 물체들을 반환합니다."""
        return [obj for obj in self._objects.values() if obj.model_type == model_type and obj.supported]
    
    def add_object(self, detection_object: DetectionObject) -> None:
        """새로운 감지 물체를 추가합니다."""
        self._objects[detection_object.name] = detection_object
    
    def remove_object(self, object_name: str) -> None:
        """물체를 제거합니다."""
        if object_name in self._objects:
            del self._objects[object_name]


# 전역 인스턴스
detection_objects_config = DetectionObjectsConfig() 