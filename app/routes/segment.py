from fastapi import APIRouter

router = APIRouter()

@router.get("/customer_segments")
def get_segments():
    return {"message": "Hello customer_segments"}
