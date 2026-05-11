from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.model.preprocessing import holistic_context
from app.model.service import model_service


router = APIRouter(tags=["translate"])


@router.websocket("/translate/stream")
async def translate_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = model_service.create_session()

    try:
        with holistic_context() as holistic:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type", "frame")

                if message_type == "reset":
                    session.reset()
                    await websocket.send_json(
                        {
                            "type": "reset",
                            "sentence": [],
                            "text": "",
                            "status": "traducción reiniciada",
                        }
                    )
                    continue

                if message_type != "frame" or not message.get("frame"):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "status": "mensaje inválido",
                            "error": "Expected a frame message with a frame payload",
                        }
                    )
                    continue

                try:
                    response = model_service.predict_frame(
                        frame_data=message["frame"],
                        session=session,
                        holistic=holistic,
                    )
                    await websocket.send_json(response.model_dump())
                except Exception as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "status": "no se pudo procesar el cuadro",
                            "error": str(exc),
                        }
                    )
    except WebSocketDisconnect:
        return
