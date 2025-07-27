FROM python:3.10-bullseye

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    cmake \
    git \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설치
RUN pip install poetry==1.8.2
RUN poetry config virtualenvs.create false

# 종속성 복사
COPY pyproject.toml ./

# MediaPipe 없는 상태로 lock 파일 생성
RUN poetry lock --no-update

# MediaPipe를 pip로 직접 설치 (ARM64 호환성)
RUN pip install mediapipe==0.10.7

# 나머지 의존성 설치
RUN poetry install --only=main --no-interaction

# 앱 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행 (로컬과 동일하게 main.py 실행)
CMD ["poetry", "run", "python", "main.py"]
