from fastapi import APIRouter, HTTPException
from app.models.schemas import DetectionRequest, DetectionResponse, DetectionResult
from app.services.detection_service import DetectionService, TRANSFORMERS_AVAILABLE
from app.core.config import settings
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config.detection_objects import detection_objects_config

router = APIRouter()
detection_service = DetectionService(yolo_model_type=settings.yolo_model_type)


@router.post("/detect", response_model=DetectionResponse)
async def detect_object(request: DetectionRequest):
    try:
        detected, confidence, bounding_box, landmarks = detection_service.detect_object_from_url(
            str(request.image_url), request.object_type, request.confidence_threshold
        )
        
        result = DetectionResult(
            detected=detected,
            confidence=confidence,
            bounding_box=bounding_box,
            landmarks=landmarks
        )
        
        message = f"{request.object_type}이(가) 감지되었습니다" if detected else f"{request.object_type}이(가) 감지되지 않았습니다"
        
        return DetectionResponse(success=True, result=result, message=message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"감지 중 오류가 발생했습니다: {str(e)}")


@router.get("/test-transformer")
async def test_transformer_model():
    """Transformer 모델 테스트용 엔드포인트"""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return {
                "success": False,
                "message": "Transformers 라이브러리를 사용할 수 없습니다.",
                "transformers_available": False
            }
        
        if detection_service.transformer_pipeline is None:
            return {
                "success": False,
                "message": "Transformer 모델이 초기화되지 않았습니다.",
                "transformers_available": True,
                "model_loaded": False
            }
        
        # 간단한 테스트 이미지 생성 (1x1 픽셀)
        import numpy as np
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # handbag 클래스로 테스트
        detected, confidence, bbox, landmarks = detection_service.detect_with_transformer(test_image, 31, 0.1)
        
        return {
            "success": True,
            "message": "Transformer 모델 테스트 완료",
            "transformers_available": True,
            "model_loaded": True,
            "test_result": {
                "detected": detected,
                "confidence": confidence,
                "bounding_box": bbox,
                "landmarks": landmarks
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Transformer 모델 테스트 중 오류: {str(e)}",
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "model_loaded": detection_service.transformer_pipeline is not None,
            "error": str(e)
        }


@router.get("/objects")
async def get_supported_objects():
    """지원되는 물체 목록을 반환합니다."""
    try:
        objects = detection_objects_config.get_all_objects()
        return {
            "success": True,
            "objects": [
                {
                    "name": obj.name,
                    "model_type": obj.model_type,
                    "model_name": obj.model_name,
                    "description": obj.description,
                    "supported": obj.supported,
                    "coco_class_id": obj.coco_class_id
                }
                for obj in objects.values()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"물체 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/objects/yolo")
async def get_yolo_objects():
    """YOLO 기반 감지 객체 목록을 반환합니다."""
    try:
        yolo_objects = detection_objects_config.get_yolo_objects()
        return {
            "success": True,
            "yolo_model_type": settings.yolo_model_type,
            "objects": [
                {
                    "name": obj.name,
                    "model_name": obj.model_name,
                    "description": obj.description,
                    "supported": obj.supported
                }
                for obj in yolo_objects
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO 물체 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/objects/transformer")
async def get_transformer_objects():
    """Transformer 기반 감지 객체 목록을 반환합니다."""
    try:
        transformer_objects = detection_objects_config.get_transformer_objects()
        return {
            "success": True,
            "transformer_model": "facebook/detr-resnet-50",
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "objects": [
                {
                    "name": obj.name,
                    "model_name": obj.model_name,
                    "description": obj.description,
                    "supported": obj.supported,
                    "coco_class_id": obj.coco_class_id
                }
                for obj in transformer_objects
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformer 물체 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/objects/mediapipe")
async def get_mediapipe_objects():
    """MediaPipe 기반 감지 객체 목록을 반환합니다."""
    try:
        mediapipe_objects = detection_objects_config.get_mediapipe_objects()
        return {
            "success": True,
            "objects": [
                {
                    "name": obj.name,
                    "model_name": obj.model_name,
                    "description": obj.description,
                    "supported": obj.supported
                }
                for obj in mediapipe_objects
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MediaPipe 물체 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "object-detection"} 