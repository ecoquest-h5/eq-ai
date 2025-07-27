# EcoQuest AI API

MediaPipe와 OpenCV를 사용한 물체 감지 API

## 설치 및 실행

```bash
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API 엔드포인트

| 메서드 | 엔드포인트                  | 설명                   |
| ------ | --------------------------- | ---------------------- |
| POST   | `/api/v1/detection/detect`  | 이미지 URL로 물체 감지 |
| GET    | `/api/v1/detection/objects` | 지원 물체 목록 조회    |
| GET    | `/health`                   | 헬스 체크              |

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

## 지원 물체

| 물체   | 모델                     | 설명        |
| ------ | ------------------------ | ----------- |
| hand   | MediaPipe Hands          | 손 감지     |
| face   | MediaPipe Face Detection | 얼굴 감지   |
| pose   | MediaPipe Pose           | 자세 감지   |
| chair  | MediaPipe Objectron      | 의자 감지   |
| cup    | MediaPipe Objectron      | 컵 감지     |
| camera | MediaPipe Objectron      | 카메라 감지 |
| shoe   | MediaPipe Objectron      | 신발 감지   |
| bottle | MediaPipe Objectron      | 병 감지     |
| bowl   | MediaPipe Objectron      | 그릇 감지   |

## 프로젝트 구조

```
ai/
├── app/                    # 애플리케이션 코드
│   ├── api/               # API 라우터
│   ├── core/              # 설정
│   ├── models/            # 데이터 모델
│   └── services/          # 비즈니스 로직
├── config/                # 설정 파일
│   └── detection_objects.py
└── tests/                 # 테스트
```
