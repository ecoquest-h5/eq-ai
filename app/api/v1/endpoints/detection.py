from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schemas import DetectionRequest, DetectionResponse, DetectionResult
from app.services.detection_service import DetectionService
from typing import Optional

router = APIRouter()
detection_service = DetectionService()


@router.post("/detect", response_model=DetectionResponse)
async def detect_object(
    file: UploadFile = File(...),
    object_type: str = Form(...),
    confidence_threshold: Optional[float] = Form(0.5)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    try:
        image_bytes = await file.read()
        detected, confidence, bounding_box, landmarks = detection_service.detect_object(
            image_bytes, object_type, confidence_threshold
        )
        
        result = DetectionResult(
            detected=detected,
            confidence=confidence,
            bounding_box=bounding_box,
            landmarks=landmarks
        )
        
        message = f"{object_type}이(가) 감지되었습니다" if detected else f"{object_type}이(가) 감지되지 않았습니다"
        
        return DetectionResponse(
            success=True,
            result=result,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"감지 중 오류가 발생했습니다: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "object-detection"} 