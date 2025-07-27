import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.services.detection_service import DetectionService
from config.detection_objects import detection_objects_config


@pytest.fixture
def detection_service():
    return DetectionService()


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


def test_config_initialization():
    """config가 올바르게 초기화되었는지 테스트"""
    config = detection_objects_config
    assert config is not None
    
    supported_objects = config.get_supported_objects()
    assert 'hand' in supported_objects
    assert 'face' in supported_objects
    assert 'pose' in supported_objects
    assert 'chair' in supported_objects


def test_get_object():
    """물체 정보 가져오기 테스트"""
    config = detection_objects_config
    
    hand_obj = config.get_object('hand')
    assert hand_obj.name == 'hand'
    assert hand_obj.model_type == 'hands'
    assert hand_obj.model_name == 'Hands'
    
    with pytest.raises(ValueError, match="지원하지 않는 물체 타입입니다"):
        config.get_object('nonexistent')


def test_get_objects_by_type():
    """모델 타입별 물체 조회 테스트"""
    config = detection_objects_config
    
    objectron_objects = config.get_objects_by_type('objectron')
    assert len(objectron_objects) > 0
    for obj in objectron_objects:
        assert obj.model_type == 'objectron'


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