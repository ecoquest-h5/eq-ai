from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DetectionObject:
    name: str
    model_type: str
    model_name: str
    description: str
    supported: bool = True
    coco_class_id: int = None  # COCO 클래스 ID (transformer 모델용)


class DetectionObjectsConfig:
    """감지 가능한 물체들의 설정을 관리합니다."""
    
    def __init__(self):
        self._objects = self._initialize_objects()
    
    def _initialize_objects(self) -> Dict[str, DetectionObject]:
        """감지 가능한 물체들을 초기화합니다."""
        """
        모델별 감지 객체들:
        
        YOLO v8 기반:
        1. bottle - 병
        2. bicycle - 자전거
        3. bus, train - 대중교통
        4. cup - 컵
        5. chair - 의자
        6. camera - 카메라
        7. shoe - 신발
        8. bowl - 그릇
        
        Transformer 기반 (facebook/detr-resnet-50):
        1. handbag - 에코백 (COCO 클래스: handbag)
        2. trash_bin - 쓰레기통 (COCO 클래스: tv)
        3. clothes - 옷감 (COCO 클래스: tie)
        
        MediaPipe 기반:
        1. hand - 손 감지
        2. face - 얼굴 감지
        3. pose - 자세 감지
        """
        return {
            # YOLO v8 기반 객체들
            'bottle': DetectionObject('bottle', 'transformer', 'bottle/vase', '병 감지 (YOLO v8)', coco_class_id=40),
            'bicycle': DetectionObject('bicycle', 'yolo', 'bicycle', '자전거 감지 (YOLO v8)'),
            'bus': DetectionObject('bus', 'yolo', 'bus/train', '버스 감지 (YOLO v8)'),
            'train': DetectionObject('train', 'yolo', 'bus/train', '기차 감지 (YOLO v8)'),
            'cup': DetectionObject('cup', 'yolo', 'cup', '컵 감지 (YOLO v8)'),
            'chair': DetectionObject('chair', 'yolo', 'chair', '의자 감지 (YOLO v8)'),
            'camera': DetectionObject('camera', 'yolo', 'cell phone', '카메라 감지 (YOLO v8)'),
            'shoe': DetectionObject('shoe', 'yolo', 'shoe', '신발 감지 (YOLO v8)'),
            'bowl': DetectionObject('bowl', 'yolo', 'bowl', '그릇 감지 (YOLO v8)'),
            
            # Transformer 기반 객체들 (facebook/detr-resnet-50)
            'handbag': DetectionObject('handbag', 'transformer', 'handbag', '에코백 감지 (DETR)', coco_class_id=31),
            'trash_bin': DetectionObject('trash_bin', 'transformer', 'tv/monitor/sink/bottle/book', '쓰레기통/분리수거함 감지 (DETR) - 전자기기, 용기, 재활용품 등 인식', coco_class_id=72),
            'clothes': DetectionObject('clothes', 'transformer', 'person/umbrella', '옷감 감지 (DETR) - person 또는 umbrella 인식', coco_class_id=27),
            
            # MediaPipe 기반 객체들
            'hand': DetectionObject('hand', 'mediapipe', 'hands', '손 감지 (MediaPipe Hands)'),
            'face': DetectionObject('face', 'mediapipe', 'face_detection', '얼굴 감지 (MediaPipe Face Detection)'),
            'pose': DetectionObject('pose', 'mediapipe', 'pose', '자세 감지 (MediaPipe Pose)'),
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
    
    def get_yolo_objects(self) -> List[DetectionObject]:
        """YOLO 기반 감지 객체들을 반환합니다."""
        return self.get_objects_by_type('yolo')
    
    def get_transformer_objects(self) -> List[DetectionObject]:
        """Transformer 기반 감지 객체들을 반환합니다."""
        return self.get_objects_by_type('transformer')
    
    def get_mediapipe_objects(self) -> List[DetectionObject]:
        """MediaPipe 기반 감지 객체들을 반환합니다."""
        return self.get_objects_by_type('mediapipe')
    
    def add_object(self, detection_object: DetectionObject) -> None:
        """새로운 감지 물체를 추가합니다."""
        self._objects[detection_object.name] = detection_object
    
    def remove_object(self, object_name: str) -> None:
        """물체를 제거합니다."""
        if object_name in self._objects:
            del self._objects[object_name]


# 전역 인스턴스
detection_objects_config = DetectionObjectsConfig() 