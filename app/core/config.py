from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "EcoQuest AI API"
    version: str = "0.0.1alpha"
    debug: bool = False
    
    class Config:
        env_file = ".env"


settings = Settings() 