from fastapi import APIRouter

from app.model.schemas import ModelInfoResponse
from app.model.service import model_service


router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return model_service.model_info()
