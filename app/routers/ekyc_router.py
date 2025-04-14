from fastapi import APIRouter

# ekyc router
ekyc_router = APIRouter()

@ekyc_router.post("/")
async def get():
    return {}