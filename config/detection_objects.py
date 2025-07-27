from typing import Dict, List
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
            'hand': DetectionObject('hand', 'hands', 'Hands', '손 감지 (MediaPipe Hands)'),
            'face': DetectionObject('face', 'face_detection', 'FaceDetection', '얼굴 감지 (MediaPipe Face Detection)'),
            'pose': DetectionObject('pose', 'pose', 'Pose', '자세 감지 (MediaPipe Pose)'),
            'chair': DetectionObject('chair', 'objectron', 'Chair', '의자 감지 (MediaPipe Objectron)'),
            'cup': DetectionObject('cup', 'objectron', 'Cup', '컵 감지 (MediaPipe Objectron)'),
            'camera': DetectionObject('camera', 'objectron', 'Camera', '카메라 감지 (MediaPipe Objectron)'),
            'shoe': DetectionObject('shoe', 'objectron', 'Shoe', '신발 감지 (MediaPipe Objectron)'),
            'bottle': DetectionObject('bottle', 'objectron', 'Bottle', '병 감지 (MediaPipe Objectron)'),
            'bowl': DetectionObject('bowl', 'objectron', 'Bowl', '그릇 감지 (MediaPipe Objectron)')
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