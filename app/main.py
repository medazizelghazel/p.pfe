from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ensure_directories
from app.database import Base, engine
from app import models
from app.api.routes.health_routes import router as health_router
from app.api.routes.auth_routes import router as auth_router
from app.api.routes.analysis_routes import router as analysis_router
from app.api.routes.admin_routes import router as admin_router
from app.api.routes.analytics_routes import router as analytics_router
from app.api.routes.platform_analytics_routes import router as platform_analytics_router


ensure_directories()

app = FastAPI(
    title="Course Analysis API",
    description="Backend API for automatic analysis of course videos.",
    version="1.0.0",
)

origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://localhost:4201",
    "http://127.0.0.1:4201",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(admin_router)
app.include_router(analytics_router)
app.include_router(platform_analytics_router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {
        "message": "Course Analysis API",
        "docs": "/docs",
        "health": "/api/health",
    }