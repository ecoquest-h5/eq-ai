import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.detection_service import DetectionService


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