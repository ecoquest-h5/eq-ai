from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "EcoQuest AI API"
    version: str = "0.0.1alpha"
    debug: bool = True
    
    # YOLO v8 모델 타입 설정 (s: small, m: medium)
    yolo_model_type: str = "s"
    
    class Config:
        env_file = ".env"


settings = Settings() 