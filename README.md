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
  - 이미지 파일 업로드
  - 지원하는 물체 타입: `hand`, `face`, `pose`, `chair`, `cup`, `camera`, `shoe`

### 헬스 체크

- **GET** `/health`
- **GET** `/api/v1/detection/health`

## 사용 예시

```bash
curl -X POST "http://localhost:8000/api/v1/detection/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "object_type=hand" \
  -F "confidence_threshold=0.5"
```
