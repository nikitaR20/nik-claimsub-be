from fastapi import APIRouter

router = APIRouter()

@router.post("/claims")
def create_claim():
    return {"message": "Hello claims"}

@router.get("/claims/{claim_id}")
def get_claim(claim_id: int):
    return {"message": f"Hello claims {claim_id}"}
