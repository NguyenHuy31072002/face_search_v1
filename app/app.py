from fastapi import FastAPI
# Define route
from app.routers.development import (basic_ekyc_router,
                                  preview_router)
# Define startup
from .startup import init_models
# Components
import time

# Tags
tags_metadata = [
    {
        "name": "Development",
        "description": "Contain features such as face alignment, face comparison for development step",
    }
]
# Define app
app = FastAPI(openapi_tags = tags_metadata)
# Add preview router
app.include_router(preview_router,
                   prefix = "/development",
                   tags = [tags_metadata[0].get("name")])
# Add basic ekyc router
app.include_router(basic_ekyc_router,
                   prefix = "/development",
                   tags = [tags_metadata[0].get("name")])

@app.on_event("startup")
async def startup_event():
    # Start
    start = time.perf_counter()
    init_models()
    print(f"Start up done after: {round(time.perf_counter() - start,1)}s")
    # Logging
    # SystemLogger.info(f"Start up done after: {round(time.perf_counter() - start,1)}s")