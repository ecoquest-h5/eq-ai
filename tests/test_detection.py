import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.services.detection_service import DetectionService, TRANSFORMERS_AVAILABLE
from config.detection_objects import detection_objects_config


@pytest.fixture
def detection_service():
    return DetectionService(yolo_model_type="s")


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_bytes():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8).tobytes()


def test_detection_service_initialization(detection_service):
    assert detection_service is not None
    assert hasattr(detection_service, 'mp_hands')
    assert hasattr(detection_service, 'mp_face')
    assert hasattr(detection_service, 'mp_pose')
    assert hasattr(detection_service, 'mp_objectron')
    assert hasattr(detection_service, 'config')
    assert hasattr(detection_service, 'yolo_model_type')
    assert hasattr(detection_service, 'transformer_pipeline')
    assert detection_service.yolo_model_type == "s"


def test_config_initialization():
    """config가 올바르게 초기화되었는지 테스트"""
    config = detection_objects_config
    assert config is not None
    
    supported_objects = config.get_supported_objects()
    assert 'hand' in supported_objects
    assert 'face' in supported_objects
    assert 'pose' in supported_objects
    assert 'bottle' in supported_objects
    assert 'bicycle' in supported_objects
    assert 'handbag' in supported_objects
    assert 'trash_bin' in supported_objects
    assert 'clothes' in supported_objects


def test_get_object():
    """물체 정보 가져오기 테스트"""
    config = detection_objects_config
    
    hand_obj = config.get_object('hand')
    assert hand_obj.name == 'hand'
    assert hand_obj.model_type == 'mediapipe'
    assert hand_obj.model_name == 'hands'
    
    bottle_obj = config.get_object('bottle')
    assert bottle_obj.name == 'bottle'
    assert bottle_obj.model_type == 'yolo'
    assert bottle_obj.model_name == 'bottle'
    
    handbag_obj = config.get_object('handbag')
    assert handbag_obj.name == 'handbag'
    assert handbag_obj.model_type == 'transformer'
    assert handbag_obj.model_name == 'handbag'
    assert handbag_obj.coco_class_id == 31
    
    clothes_obj = config.get_object('clothes')
    assert clothes_obj.name == 'clothes'
    assert clothes_obj.model_type == 'transformer'
    assert clothes_obj.model_name == 'person/umbrella'
    assert clothes_obj.coco_class_id == 27
    
    with pytest.raises(ValueError, match="지원하지 않는 물체 타입입니다"):
        config.get_object('nonexistent')


def test_get_objects_by_type():
    """모델 타입별 물체 조회 테스트"""
    config = detection_objects_config
    
    yolo_objects = config.get_objects_by_type('yolo')
    assert len(yolo_objects) > 0
    for obj in yolo_objects:
        assert obj.model_type == 'yolo'
    
    transformer_objects = config.get_objects_by_type('transformer')
    assert len(transformer_objects) > 0
    for obj in transformer_objects:
        assert obj.model_type == 'transformer'
    
    mediapipe_objects = config.get_objects_by_type('mediapipe')
    assert len(mediapipe_objects) > 0
    for obj in mediapipe_objects:
        assert obj.model_type == 'mediapipe'


def test_get_yolo_objects():
    """YOLO 기반 객체 조회 테스트"""
    config = detection_objects_config
    yolo_objects = config.get_yolo_objects()
    
    assert len(yolo_objects) > 0
    yolo_object_names = [obj.name for obj in yolo_objects]
    assert 'bottle' in yolo_object_names
    assert 'bicycle' in yolo_object_names
    assert 'handbag' not in yolo_object_names  # handbag은 transformer


def test_get_transformer_objects():
    """Transformer 기반 객체 조회 테스트"""
    config = detection_objects_config
    transformer_objects = config.get_transformer_objects()
    
    assert len(transformer_objects) > 0
    transformer_object_names = [obj.name for obj in transformer_objects]
    assert 'handbag' in transformer_object_names
    assert 'trash_bin' in transformer_object_names
    assert 'clothes' in transformer_object_names
    assert 'bottle' not in transformer_object_names  # bottle은 yolo


def test_get_mediapipe_objects():
    """MediaPipe 기반 객체 조회 테스트"""
    config = detection_objects_config
    mediapipe_objects = config.get_mediapipe_objects()
    
    assert len(mediapipe_objects) > 0
    mediapipe_object_names = [obj.name for obj in mediapipe_objects]
    assert 'hand' in mediapipe_object_names
    assert 'face' in mediapipe_object_names
    assert 'pose' in mediapipe_object_names


@patch('requests.get')
def test_download_image_from_url_success(mock_get, detection_service, sample_image_bytes):
    mock_response = MagicMock()
    mock_response.content = sample_image_bytes
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    
    result = detection_service.download_image_from_url("https://example.com/image.jpg")
    assert result == sample_image_bytes


@patch('requests.get')
def test_download_image_from_url_failure(mock_get, detection_service):
    mock_get.side_effect = Exception("Network error")
    
    with pytest.raises(Exception, match="이미지 다운로드 실패"):
        detection_service.download_image_from_url("https://example.com/image.jpg")


def test_detect_hands_no_hands(detection_service, sample_image):
    detected, confidence, bbox, landmarks = detection_service.detect_hands(sample_image)
    assert detected is False
    assert confidence == 0.0


def test_detect_faces_no_faces(detection_service, sample_image):
    detected, confidence, bbox, landmarks = detection_service.detect_faces(sample_image)
    assert detected is False
    assert confidence == 0.0


def test_detect_pose_no_pose(detection_service, sample_image):
    detected, confidence, bbox, landmarks = detection_service.detect_pose(sample_image)
    assert detected is False
    assert confidence == 0.0


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_with_transformer_success(mock_pipeline, detection_service, sample_image):
    """Transformer 감지 성공 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label_id': 31,  # handbag class
            'score': 0.85,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 31, 0.5)
    assert detected is True
    assert confidence == 0.85
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_with_transformer_no_detection(mock_pipeline, detection_service, sample_image):
    """Transformer 감지 실패 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = []  # 감지 없음
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 31, 0.5)
    assert detected is False
    assert confidence == 0.0


def test_detect_with_transformer_not_available(detection_service, sample_image):
    """Transformers 라이브러리를 사용할 수 없을 때 테스트"""
    if not TRANSFORMERS_AVAILABLE:
        with pytest.raises(Exception, match="Transformers 라이브러리를 사용할 수 없습니다"):
            detection_service.detect_with_transformer(sample_image, 31, 0.5)


@patch('ultralytics.YOLO')
def test_detect_with_yolo_success(mock_yolo, detection_service, sample_image):
    """YOLO 감지 성공 테스트"""
    # Mock YOLO 모델 설정
    mock_model = MagicMock()
    mock_model.names = {0: 'bottle', 1: 'cup', 2: 'chair'}
    mock_model.return_value = [MagicMock()]
    
    # Mock 결과 설정
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [0]  # bottle class
    mock_box.conf = [0.8]  # confidence
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 200]
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]
    
    detection_service.yolo_model = mock_model
    
    detected, confidence, bbox, landmarks = detection_service.detect_with_yolo(sample_image, 'bottle', 0.5)
    assert detected is True
    assert confidence == 0.8
    assert bbox is not None
    assert landmarks is not None


@patch('ultralytics.YOLO')
def test_detect_with_yolo_no_detection(mock_yolo, detection_service, sample_image):
    """YOLO 감지 실패 테스트"""
    # Mock YOLO 모델 설정
    mock_model = MagicMock()
    mock_model.names = {0: 'bottle', 1: 'cup', 2: 'chair'}
    mock_model.return_value = [MagicMock()]
    
    # Mock 결과 설정 (감지 없음)
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model.return_value = [mock_result]
    
    detection_service.yolo_model = mock_model
    
    detected, confidence, bbox, landmarks = detection_service.detect_with_yolo(sample_image, 'bottle', 0.5)
    assert detected is False
    assert confidence == 0.0


def test_detect_objects_invalid_type(detection_service, sample_image):
    """잘못된 모델 타입으로 objectron 감지 시도"""
    with pytest.raises(ValueError, match="잘못된 모델 타입입니다"):
        detection_service.detect_objects(sample_image, 'hand', 0.5)


def test_detect_object_from_url_invalid_type(detection_service):
    """존재하지 않는 물체 타입으로 감지 시도"""
    with patch.object(detection_service, 'process_image_from_url') as mock_process:
        mock_process.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="지원하지 않는 물체 타입입니다"):
            detection_service.detect_object_from_url("https://example.com/image.jpg", "nonexistent", 0.5)


def test_detect_object_from_url_yolo_type(detection_service):
    """YOLO 타입 객체 감지 테스트"""
    with patch.object(detection_service, 'process_image_from_url') as mock_process:
        mock_process.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch.object(detection_service, 'detect_with_yolo') as mock_yolo:
            mock_yolo.return_value = (True, 0.8, [100, 100, 100, 100], [[100, 100], [200, 100], [200, 200], [100, 200]])
            
            detected, confidence, bbox, landmarks = detection_service.detect_object_from_url("https://example.com/image.jpg", "bottle", 0.5)
            
            assert detected is True
            assert confidence == 0.8
            mock_yolo.assert_called_once()


def test_detect_object_from_url_transformer_type(detection_service):
    """Transformer 타입 객체 감지 테스트"""
    with patch.object(detection_service, 'process_image_from_url') as mock_process:
        mock_process.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        if TRANSFORMERS_AVAILABLE:
            with patch.object(detection_service, 'detect_with_transformer') as mock_transformer:
                mock_transformer.return_value = (True, 0.85, [100, 100, 100, 100], [[100, 100], [200, 100], [200, 200], [100, 200]])
                
                detected, confidence, bbox, landmarks = detection_service.detect_object_from_url("https://example.com/image.jpg", "handbag", 0.5)
                
                assert detected is True
                assert confidence == 0.85
                mock_transformer.assert_called_once()
        else:
            with pytest.raises(Exception, match="Transformers 라이브러리를 사용할 수 없어"):
                detection_service.detect_object_from_url("https://example.com/image.jpg", "handbag", 0.5) 


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_clothes_with_person(mock_pipeline, detection_service, sample_image):
    """clothes 감지 - person 인식 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'person',
            'score': 0.85,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # clothes 클래스 ID (27)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 27, 0.5)
    assert detected is True
    assert confidence == 0.85
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_clothes_with_umbrella(mock_pipeline, detection_service, sample_image):
    """clothes 감지 - umbrella 인식 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'umbrella',
            'score': 0.75,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # clothes 클래스 ID (27)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 27, 0.5)
    assert detected is True
    assert confidence == 0.75
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_clothes_with_other_object(mock_pipeline, detection_service, sample_image):
    """clothes 감지 - 다른 객체 인식 시 실패 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'car',
            'score': 0.9,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # clothes 클래스 ID (27)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 27, 0.5)
    assert detected is False
    assert confidence == 0.0 


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_tv(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - tv 인식 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'tv',
            'score': 0.8,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is True
    assert confidence == 0.8
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_monitor(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - monitor 인식 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'monitor',
            'score': 0.7,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is True
    assert confidence == 0.7
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_box(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - box 인식 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'box',
            'score': 0.6,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is True
    assert confidence == 0.6
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_other_object(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - 다른 객체 인식 시 실패 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'car',
            'score': 0.9,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is False
    assert confidence == 0.0 


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_sink(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - sink 인식 테스트"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'sink',
            'score': 0.8,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is True
    assert confidence == 0.8
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_bottle(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - bottle 인식 테스트 (재활용 관련)"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'bottle',
            'score': 0.7,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is True
    assert confidence == 0.7
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_book(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - book 인식 테스트 (재활용 관련)"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'book',
            'score': 0.6,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is True
    assert confidence == 0.6
    assert bbox is not None
    assert landmarks is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers 라이브러리를 사용할 수 없습니다")
@patch('transformers.pipeline')
def test_detect_trash_bin_with_cell_phone(mock_pipeline, detection_service, sample_image):
    """trash_bin 감지 - cell phone 인식 테스트 (재활용 관련)"""
    # Mock Transformer pipeline 설정
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {
            'label': 'cell phone',
            'score': 0.9,
            'box': {'xmin': 100, 'ymin': 100, 'xmax': 200, 'ymax': 200}
        }
    ]
    mock_pipeline.return_value = mock_pipe
    
    detection_service.transformer_pipeline = mock_pipe
    
    # trash_bin 클래스 ID (72)로 테스트
    detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(sample_image, 72, 0.5)
    assert detected is True
    assert confidence == 0.9
    assert bbox is not None
    assert landmarks is not None 