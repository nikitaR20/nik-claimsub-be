from fastapi import APIRouter

router = APIRouter()

@router.get("/providers")
def get_providers():
    return {"message": "Hello providers"}
