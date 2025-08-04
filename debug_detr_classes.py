#!/usr/bin/env python3
"""
DETR 모델 클래스 확인 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from transformers import pipeline
    import torch
    from PIL import Image
    import numpy as np
    
    print("=== DETR 모델 클래스 확인 ===")
    
    # DETR 모델 초기화
    device = 0 if torch.cuda.is_available() else -1
    print(f"디바이스: {'GPU' if device == 0 else 'CPU'}")
    
    pipeline_obj = pipeline('object-detection', model='facebook/detr-resnet-50', device=device)
    
    # 간단한 테스트 이미지 생성
    test_image = Image.fromarray(np.ones((200, 200, 3), dtype=np.uint8) * 128)
    
    # 예측 실행
    results = pipeline_obj(test_image)
    
    print(f"\n감지된 객체 수: {len(results)}")
    
    if results:
        print("\n감지된 객체들:")
        for i, result in enumerate(results):
            label = result.get('label', 'unknown')
            score = result.get('score', 0.0)
            print(f"  {i+1}. {label} (신뢰도: {score:.3f})")
    
    # COCO 클래스 목록 확인 (일부)
    print("\n=== COCO 클래스 일부 ===")
    coco_classes = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
        15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow',
        21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack',
        26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee',
        31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
        36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
        40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife',
        45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
        50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
        55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant',
        60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop',
        65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave',
        70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book',
        75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
        80: 'toothbrush'
    }
    
    # trash_bin 관련 클래스들
    print("\n=== trash_bin 관련 클래스들 ===")
    trash_related = {k: v for k, v in coco_classes.items() if any(word in v.lower() for word in ['tv', 'monitor', 'screen', 'box', 'container', 'sink'])}
    for class_id, class_name in trash_related.items():
        print(f"  {class_id}: {class_name}")
    
    # clothes 관련 클래스들
    print("\n=== clothes 관련 클래스들 ===")
    clothes_related = {k: v for k, v in coco_classes.items() if any(word in v.lower() for word in ['person', 'umbrella', 'backpack', 'handbag', 'tie', 'suitcase'])}
    for class_id, class_name in clothes_related.items():
        print(f"  {class_id}: {class_name}")
    
except ImportError as e:
    print(f"Transformers 라이브러리 로드 실패: {e}")
except Exception as e:
    print(f"오류 발생: {e}") 