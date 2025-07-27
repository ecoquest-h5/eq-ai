import pytest
import numpy as np
from app.services.detection_service import DetectionService


@pytest.fixture
def detection_service():
    return DetectionService()


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_detection_service_initialization(detection_service):
    assert detection_service is not None
    assert hasattr(detection_service, 'mp_hands')
    assert hasattr(detection_service, 'mp_face')
    assert hasattr(detection_service, 'mp_pose')
    assert hasattr(detection_service, 'mp_objectron')


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