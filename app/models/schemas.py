from pydantic import BaseModel
from typing import List, Optional


class DetectionRequest(BaseModel):
    object_type: str
    confidence_threshold: Optional[float] = 0.5


class DetectionResult(BaseModel):
    detected: bool
    confidence: float
    bounding_box: Optional[List[float]] = None
    landmarks: Optional[List[List[float]]] = None


class DetectionResponse(BaseModel):
    success: bool
    result: DetectionResult
    message: str 