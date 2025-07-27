from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.api import api_router
import uvicorn

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "EcoQuest AI API", "version": settings.version}


@app.get("/health")
async def health_check():
    return {"status": "healthy"} 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)