#!/usr/bin/env python3
"""
Transformer 모델 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app.services.detection_service import DetectionService, TRANSFORMERS_AVAILABLE
import numpy as np

def test_transformer():
    print("=== Transformer 모델 테스트 ===")
    
    # DetectionService 초기화
    service = DetectionService(yolo_model_type="s")
    
    print(f"Transformers 사용 가능: {TRANSFORMERS_AVAILABLE}")
    print(f"Transformer 모델 로드됨: {service.transformer_pipeline is not None}")
    
    if not TRANSFORMERS_AVAILABLE or service.transformer_pipeline is None:
        print("Transformer 모델을 사용할 수 없습니다.")
        return
    
    # 테스트 이미지 생성 (회색 이미지)
    test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    print("\n=== handbag 감지 테스트 ===")
    try:
        detected, confidence, bbox, landmarks = service.detect_with_transformer(test_image, 31, 0.1)
        print(f"감지됨: {detected}")
        print(f"신뢰도: {confidence}")
        print(f"바운딩 박스: {bbox}")
        print(f"랜드마크: {landmarks}")
    except Exception as e:
        print(f"오류 발생: {e}")
    
    print("\n=== trash_bin 감지 테스트 (분리수거/재활용 관련) ===")
    try:
        detected, confidence, bbox, landmarks = service.detect_with_transformer(test_image, 72, 0.1)
        print(f"감지됨: {detected}")
        print(f"신뢰도: {confidence}")
        print(f"바운딩 박스: {bbox}")
        print(f"랜드마크: {landmarks}")
        print("참고: trash_bin은 전자기기, 용기, 재활용품 등이 감지되면 정답으로 처리됩니다.")
        print("포함 클래스: tv, monitor, sink, bottle, book, cell phone, keyboard, mouse 등")
    except Exception as e:
        print(f"오류 발생: {e}")
    
    print("\n=== clothes 감지 테스트 (person/umbrella) ===")
    try:
        detected, confidence, bbox, landmarks = service.detect_with_transformer(test_image, 27, 0.1)
        print(f"감지됨: {detected}")
        print(f"신뢰도: {confidence}")
        print(f"바운딩 박스: {bbox}")
        print(f"랜드마크: {landmarks}")
        print("참고: clothes는 person 또는 umbrella가 감지되면 정답으로 처리됩니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    test_transformer() 