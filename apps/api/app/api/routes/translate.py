from typing import cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.lessa.inference import model_service
from app.lessa.preprocessing import holistic_context
from app.lessa.schemas import InferenceMode


router = APIRouter(tags=["translate"])
VALID_MODES: set[str] = {"auto", "words", "alphabet"}


@router.websocket("/translate/stream")
async def translate_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = model_service.create_session()

    try:
        # MediaPipe Holistic owns native resources; keep it scoped to the socket
        # lifetime so disconnects release camera-frame processing resources promptly.
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

                requested_mode = message.get("mode", "auto")
                if requested_mode not in VALID_MODES:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "status": "modo inválido",
                            "error": "Expected mode to be auto, words, or alphabet",
                        }
                    )
                    continue

                try:
                    response = model_service.predict_frame(
                        frame_data=message["frame"],
                        session=session,
                        holistic=holistic,
                        requested_mode=cast(InferenceMode, requested_mode),
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
