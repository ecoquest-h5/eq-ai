# EcoQuest AI API

YOLO v8, Transformer(DETR), MediaPipe를 사용한 물체 감지 API

## 설치 및 실행

### 시스템 요구사항

- Python 3.10 이상
- PyTorch 2.1.0 이상 (Transformers 라이브러리 사용을 위해)
- CUDA 지원 GPU (선택사항, Transformer 모델 가속화용)

### 로컬 실행

```bash
# 의존성 설치
poetry install

# 서버 실행
poetry run python3 main.py
```

### Docker 실행

```bash
# 프로덕션 빌드
docker build -t ecoquest-ai .
docker run -p 8000:8000 ecoquest-ai
```

## 모델 설정

### YOLO 모델 설정

`app/core/config.py` 파일에서 YOLO v8 모델 타입을 설정할 수 있습니다:

```python
class Settings(BaseSettings):
    # YOLO v8 모델 타입 설정 (s: small, m: medium)
    yolo_model_type: str = "s"
```

또는 환경 변수를 통해 설정할 수 있습니다:

```bash
export YOLO_MODEL_TYPE=m
poetry run python3 main.py
```

- `s`: YOLOv8s (작고 빠름, 정확도 보통)
- `m`: YOLOv8m (중간 크기, 정확도 높음)

### Transformer 모델 설정

기본적으로 `facebook/detr-resnet-50` 모델을 사용합니다.

**중요**: PyTorch 2.1.0 이상이 필요합니다. PyTorch 버전이 낮으면 Transformer 모델이 자동으로 비활성화됩니다.

GPU가 있다면 자동으로 감지하여 사용합니다:

```python
# 자동 GPU 감지 (GPU가 있으면 GPU, 없으면 CPU 사용)
device = 0 if torch.cuda.is_available() else -1
```

## API 엔드포인트

| 메서드 | 엔드포인트                              | 설명                            |
| ------ | --------------------------------------- | ------------------------------- |
| POST   | `/api/v1/detection/detect`              | 이미지 URL로 물체 감지          |
| GET    | `/api/v1/detection/objects`             | 모든 지원 물체 목록 조회        |
| GET    | `/api/v1/detection/objects/yolo`        | YOLO 기반 물체 목록 조회        |
| GET    | `/api/v1/detection/objects/transformer` | Transformer 기반 물체 목록 조회 |
| GET    | `/api/v1/detection/objects/mediapipe`   | MediaPipe 기반 물체 목록 조회   |
| GET    | `/config/yolo-model`                    | 현재 YOLO 모델 설정 조회        |
| GET    | `/health`                               | 헬스 체크                       |

## 사용 예시

### 물체 감지

```bash
curl -X POST "http://localhost:8000/api/v1/detection/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "object_type": "handbag",
    "confidence_threshold": 0.5
  }'
```

### YOLO 기반 물체 목록 조회

```bash
curl -X GET "http://localhost:8000/api/v1/detection/objects/yolo"
```

### Transformer 기반 물체 목록 조회

```bash
curl -X GET "http://localhost:8000/api/v1/detection/objects/transformer"
```

### YOLO 모델 설정 조회

```bash
curl -X GET "http://localhost:8000/config/yolo-model"
```

## 지원 물체

### YOLO v8 기반 객체

| 물체    | COCO 클래스 | 설명        |
| ------- | ----------- | ----------- |
| bottle  | bottle      | 병 감지     |
| bicycle | bicycle     | 자전거 감지 |
| bus     | bus         | 버스 감지   |
| train   | train       | 기차 감지   |
| cup     | cup         | 컵 감지     |
| chair   | chair       | 의자 감지   |
| camera  | cell phone  | 카메라 감지 |
| shoe    | shoe        | 신발 감지   |
| bowl    | bowl        | 그릇 감지   |

### Transformer 기반 객체 (facebook/detr-resnet-50)

**주의**: PyTorch 2.1.0 이상이 필요합니다.

| 물체      | COCO 클래스 | COCO ID | 설명          |
| --------- | ----------- | ------- | ------------- |
| handbag   | handbag     | 31      | 에코백 감지   |
| trash_bin | tv/monitor/sink/bottle/book | 72 | 쓰레기통/분리수거함 감지 (전자기기, 용기, 재활용품 등 인식) |
| clothes   | person/umbrella | 27 | 옷감 감지 (person 또는 umbrella 인식) |

### MediaPipe 기반 객체

| 물체 | 모델                     | 설명      |
| ---- | ------------------------ | --------- |
| hand | MediaPipe Hands          | 손 감지   |
| face | MediaPipe Face Detection | 얼굴 감지 |
| pose | MediaPipe Pose           | 자세 감지 |

## 프로젝트 구조

```
ai/
├── app/                    # 애플리케이션 코드
│   ├── api/               # API 라우터
│   │   └── v1/
│   │       └── endpoints/
│   │           └── detection.py
│   ├── core/              # 설정
│   │   └── config.py
│   ├── models/            # 데이터 모델
│   └── services/          # 비즈니스 로직
│       └── detection_service.py
├── config/                # 설정 파일
│   └── detection_objects.py
├── tests/                 # 테스트
│   └── test_detection.py
├── Dockerfile             # Docker 설정
├── pyproject.toml         # 의존성 관리
├── main.py                # 메인 애플리케이션
└── README.md              # 프로젝트 문서
```

## 의존성

주요 의존성:

- `ultralytics`: YOLO v8 모델
- `transformers`: Transformer 모델 (DETR) - PyTorch 2.1.0+ 필요
- `torch`: PyTorch (YOLO, DETR 백엔드) - 2.1.0+ 필요
- `mediapipe`: MediaPipe 모델들
- `opencv-python`: 이미지 처리
- `fastapi`: 웹 API 프레임워크
- `pydantic`: 데이터 검증
- `pillow`: 이미지 처리 (PIL)
- `accelerate`: Transformer 가속화

## 모델별 특징

### YOLO v8

- **장점**: 빠른 속도, 실시간 감지 가능
- **단점**: 일부 복잡한 객체 인식 정확도 낮음
- **적합한 객체**: bottle, bicycle, bus, train, cup, chair, camera, shoe, bowl

### Transformer (DETR)

- **장점**: 높은 정확도, 복잡한 객체 인식 우수
- **단점**: 상대적으로 느린 속도, PyTorch 2.1.0+ 필요
- **적합한 객체**: handbag, trash_bin, clothes
- **주의**: PyTorch 버전이 낮으면 자동으로 비활성화됨

### MediaPipe

- **장점**: 경량화, 실시간 처리
- **단점**: 제한된 객체 타입
- **적합한 객체**: hand, face, pose

## 문제 해결

### Transformer 모델이 로드되지 않는 경우

1. **PyTorch 버전 확인**:

   ```bash
   poetry run python -c "import torch; print(torch.__version__)"
   ```

2. **PyTorch 업데이트**:

   ```bash
   poetry update torch torchvision
   ```

3. **Transformers 라이브러리 재설치**:
   ```bash
   poetry remove transformers
   poetry add transformers
   ```

### GPU 사용 확인

```bash
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 테스트

```bash
# 모든 테스트 실행
poetry run pytest

# 특정 테스트 파일 실행
poetry run pytest tests/test_detection.py

# 커버리지와 함께 실행
poetry run pytest --cov=app tests/

# Transformers 관련 테스트만 실행
poetry run pytest tests/test_detection.py -k "transformer"
```
