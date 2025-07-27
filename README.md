# EcoQuest AI API

MediaPipe와 OpenCV를 사용한 물체 감지 API

## 설치

```bash
poetry install
```

## 실행

```bash
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### 물체 감지

- **POST** `/api/v1/detection/detect`
  - 이미지 URL을 body로 전송
  - 지원하는 물체 타입: `hand`, `face`, `pose`, `chair`, `cup`, `camera`, `shoe`, `bottle`, `bowl`

### 지원 물체 목록 조회

- **GET** `/api/v1/detection/objects`
  - 감지 가능한 모든 물체 목록과 정보를 반환

### 헬스 체크

- **GET** `/health`
- **GET** `/api/v1/detection/health`

## 사용 예시

### 물체 감지

```bash
curl -X POST "http://localhost:8000/api/v1/detection/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "object_type": "hand",
    "confidence_threshold": 0.5
  }'
```

### 지원 물체 목록 조회

```bash
curl -X GET "http://localhost:8000/api/v1/detection/objects"
```

## 요청 형식

```json
{
  "image_url": "https://example.com/image.jpg",
  "object_type": "hand",
  "confidence_threshold": 0.5
}
```

## 응답 형식

### 감지 결과

```json
{
  "success": true,
  "result": {
    "detected": true,
    "confidence": 0.85,
    "bounding_box": null,
    "landmarks": [[0.1, 0.2, 0.3], ...]
  },
  "message": "hand이(가) 감지되었습니다"
}
```

### 물체 목록

```json
{
  "success": true,
  "objects": [
    {
      "name": "hand",
      "model_type": "hands",
      "model_name": "Hands",
      "description": "손 감지 (MediaPipe Hands)",
      "supported": true
    }
  ]
}
```

## 프로젝트 구조

```
ai/
├── app/                    # 애플리케이션 코드
│   ├── api/               # API 라우터
│   ├── core/              # 설정 및 핵심 기능
│   ├── models/            # 데이터 모델
│   └── services/          # 비즈니스 로직
├── config/                # 설정 파일
│   └── detection_objects.py  # 감지 가능한 물체 설정
└── tests/                 # 테스트 코드
```

## 감지 가능한 물체

- **hand**: 손 감지 (MediaPipe Hands)
- **face**: 얼굴 감지 (MediaPipe Face Detection)
- **pose**: 자세 감지 (MediaPipe Pose)
- **chair**: 의자 감지 (MediaPipe Objectron)
- **cup**: 컵 감지 (MediaPipe Objectron)
- **camera**: 카메라 감지 (MediaPipe Objectron)
- **shoe**: 신발 감지 (MediaPipe Objectron)
- **bottle**: 병 감지 (MediaPipe Objectron)
- **bowl**: 그릇 감지 (MediaPipe Objectron)
