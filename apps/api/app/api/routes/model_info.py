from fastapi import APIRouter

from app.lessa.inference import model_service
from app.lessa.schemas import ModelInfoResponse


router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return model_service.model_info()
