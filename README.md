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
  - 지원하는 물체 타입: `hand`, `face`, `pose`, `chair`, `cup`, `camera`, `shoe`

### 헬스 체크

- **GET** `/health`
- **GET** `/api/v1/detection/health`

## 사용 예시

```bash
curl -X POST "http://localhost:8000/api/v1/detection/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "object_type": "hand",
    "confidence_threshold": 0.5
  }'
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
