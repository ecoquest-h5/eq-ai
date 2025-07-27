from fastapi import APIRouter, HTTPException
from app.models.schemas import DetectionRequest, DetectionResponse, DetectionResult
from app.services.detection_service import DetectionService
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config.detection_objects import detection_objects_config

router = APIRouter()
detection_service = DetectionService()


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
                    "supported": obj.supported
                }
                for obj in objects.values()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"물체 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "object-detection"} 