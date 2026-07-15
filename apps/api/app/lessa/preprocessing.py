"""Preprocesamiento de fotogramas a vectores de características para los modelos LESSA.

Este módulo es el puente entre el fotograma crudo de la cámara y la entrada numérica que
esperan los modelos entrenados. El flujo por fotograma es:

    JPEG base64 -> decode_frame -> BGR ndarray
                -> mediapipe_detection -> landmarks Holistic (pose, rostro, manos)
                -> extract_hybrid_keypoints -> vector de posición de 306 valores
                -> build_sequence_features -> secuencia con velocidad/aceleración (918 por fotograma)
                -> pad_sequence -> tensor (1, sequence_length, feature_length)

Contrato de posición (306 valores por fotograma), en este orden exacto de concatenación:
    pose (33 x 4: x, y, z, visibility) = 132
    rostro seleccionado (16 x 3: x, y, z) = 48
    mano izquierda (21 x 3) = 63
    mano derecha (21 x 3) = 63
    total = 306

Cambiar el orden, la cantidad de puntos o los índices de rostro rompe la compatibilidad con
los artefactos entrenados. Ver `docs/ARCHITECTURE.md` (sección del contrato de características).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


# Los modelos de palabras y de alfabeto se entrenaron con diferentes subconjuntos de rostro de 16 puntos.
# Mantenga estos órdenes estables: la posición en el vector de características es parte del contrato de entrada del modelo.
WORD_FACE_INDICES = [
    61,
    291,
    0,
    17,
    13,
    14,
    33,
    133,
    362,
    263,
    70,
    63,
    105,
    336,
    296,
    334,
]

ALPHABET_FACE_INDICES = [
    33,
    133,
    362,
    263,
    1,
    61,
    291,
    199,
    94,
    0,
    11,
    13,
    14,
    15,
    16,
    17,
]


@dataclass(frozen=True)
class FrameExtraction:
    word_keypoints: np.ndarray
    alphabet_keypoints: np.ndarray
    has_hands: bool


@dataclass(frozen=True)
class KeypointExtraction:
    position_keypoints: np.ndarray
    has_hands: bool


def decode_frame(frame_data: str) -> np.ndarray:
    """Decodifica un fotograma JPEG en base64 (data URL o base64 puro) a un ndarray BGR de OpenCV.

    El navegador envía cada fotograma como data URL (`data:image/jpeg;base64,...`); se descarta el
    prefijo antes de la coma. Lanza ValueError si el payload no es una imagen válida.
    """
    if "," in frame_data:
        frame_data = frame_data.split(",", 1)[1]

    raw = base64.b64decode(frame_data)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image frame payload")
    return frame


def mediapipe_detection(image: np.ndarray, holistic) -> object:
    """Ejecuta MediaPipe Holistic sobre un fotograma BGR y devuelve los landmarks detectados.

    OpenCV entrega fotogramas en BGR, pero MediaPipe espera RGB. La bandera `writeable=False`
    permite a MediaPipe evitar una copia interna del arreglo mientras procesa la imagen.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    return holistic.process(image_rgb)


def extract_hybrid_keypoints(results: object, base_feature_length: int) -> FrameExtraction:
    """Construye ambos vectores de posición (palabras y alfabeto) a partir de un mismo resultado.

    Los dos modelos comparten pose y manos pero usan subconjuntos de rostro distintos, por eso se
    extrae una vez por cada juego de índices. `has_hands` se toma del vector de palabras porque la
    detección de manos es idéntica entre ambos.
    """
    word = _extract_keypoints_for_face_indices(results, WORD_FACE_INDICES, base_feature_length)
    alphabet = _extract_keypoints_for_face_indices(results, ALPHABET_FACE_INDICES, base_feature_length)
    return FrameExtraction(
        word_keypoints=word.position_keypoints,
        alphabet_keypoints=alphabet.position_keypoints,
        has_hands=word.has_hands,
    )


def extract_keypoints(results: object, base_feature_length: int) -> KeypointExtraction:
    return _extract_keypoints_for_face_indices(results, WORD_FACE_INDICES, base_feature_length)


def _extract_keypoints_for_face_indices(
    results: object,
    face_indices: list[int],
    base_feature_length: int,
) -> KeypointExtraction:
    """Aplana los landmarks a un vector de posición de `base_feature_length` valores.

    Los landmarks faltantes (por ejemplo, una mano fuera de cuadro) se rellenan con ceros para
    conservar siempre el mismo tamaño de vector. Al final valida que la longitud coincida con el
    contrato del modelo y lanza ValueError si no es así, lo que suele indicar un cambio de índices.
    """
    # Todas las coordenadas se anclan al punto de referencia (landmark) de la nariz para que las
    # entradas del modelo representen el movimiento relativo a la persona señante en lugar de la
    # posición absoluta en la webcam.
    if results.pose_landmarks:
        anchor_x = results.pose_landmarks.landmark[0].x
        anchor_y = results.pose_landmarks.landmark[0].y
        anchor_z = results.pose_landmarks.landmark[0].z
    else:
        anchor_x, anchor_y, anchor_z = 0.0, 0.0, 0.0

    if results.pose_landmarks:
        pose = np.array(
            [
                [res.x - anchor_x, res.y - anchor_y, res.z - anchor_z, res.visibility]
                for res in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [
                [res.x - anchor_x, res.y - anchor_y, res.z - anchor_z]
                for res in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        left_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [
                [res.x - anchor_x, res.y - anchor_y, res.z - anchor_z]
                for res in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.face_landmarks:
        face = np.array(
            [
                [
                    results.face_landmarks.landmark[i].x - anchor_x,
                    results.face_landmarks.landmark[i].y - anchor_y,
                    results.face_landmarks.landmark[i].z - anchor_z,
                ]
                for i in face_indices
            ],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(len(face_indices) * 3, dtype=np.float32)

    position_keypoints = np.concatenate([pose, face, left_hand, right_hand]).astype(np.float32)
    if position_keypoints.shape[0] != base_feature_length:
        raise ValueError(
            f"Expected {base_feature_length} position features, got {position_keypoints.shape[0]}"
        )

    return KeypointExtraction(
        position_keypoints=position_keypoints,
        has_hands=bool(results.left_hand_landmarks or results.right_hand_landmarks),
    )


def build_sequence_features(
    position_sequence: list[np.ndarray],
    use_temporal_features: bool,
    temporal_delta_order: int,
) -> list[np.ndarray]:
    """Enriquece una secuencia de posiciones con velocidad y aceleración por fotograma.

    Debe reproducir exactamente `compute_deltas` del script de entrenamiento
    (research/words/train_temporal_model.py): posición (306) + velocidad (306) + aceleración (306)
    = 918 valores por fotograma. Si difiere del entrenamiento, el modelo recibe una entrada
    inconsistente. Sin características temporales devuelve solo las posiciones.
    """
    sequence = np.asarray(position_sequence, dtype=np.float32)

    if not use_temporal_features:
        return list(sequence)

    # El modelo de palabras espera las posiciones más los deltas de primer orden y, opcionalmente,
    # de segundo orden en este orden exacto de concatenación. El relleno ocurre después de los deltas
    # para que la ventana de movimiento activo produzca la misma forma de 918 características usada
    # durante el entrenamiento.
    delta = np.vstack([sequence[0:1, :], np.diff(sequence, axis=0)]).astype(np.float32)
    features = [sequence, delta]

    if temporal_delta_order >= 2:
        acceleration = np.vstack([delta[0:1, :], np.diff(delta, axis=0)]).astype(np.float32)
        features.append(acceleration)

    return list(np.concatenate(features, axis=-1).astype(np.float32))


def motion_score(position_sequence: list[np.ndarray], hand_feature_start: int = 180) -> float:
    """Mide el movimiento reciente de las manos para enrutar entre palabras y alfabeto en modo Auto.

    `hand_feature_start=180` es el índice donde empiezan las características de las manos dentro del
    vector de 306: pose (33 x 4 = 132) + rostro (16 x 3 = 48) = 180. Se ignoran pose y rostro para
    que el puntaje refleje solo el movimiento de las manos. Devuelve el promedio del valor absoluto
    de las diferencias de los últimos fotogramas: alto = seña dinámica, bajo = seña estática.
    """
    if len(position_sequence) < 2:
        return 0.0

    sequence = np.asarray(position_sequence, dtype=np.float32)
    delta = np.vstack([sequence[0:1, :], np.diff(sequence, axis=0)]).astype(np.float32)
    recent_delta = delta[-5:, hand_feature_start:]
    return float(np.mean(np.abs(recent_delta))) if recent_delta.size else 0.0


def pad_sequence(sequence: list[np.ndarray], sequence_length: int, feature_length: int) -> np.ndarray:
    """Rellena (o trunca) la secuencia a la forma fija (1, sequence_length, feature_length).

    La LSTM espera siempre `sequence_length` fotogramas. Las secuencias más cortas se rellenan con
    ceros al final (el modelo usa una capa Masking para ignorarlos); las más largas se truncan. Se
    añade una dimensión de batch al frente para llamar directamente a `model.predict`.
    """
    padded = np.zeros((sequence_length, feature_length), dtype=np.float32)
    usable = sequence[:sequence_length]
    if usable:
        padded[: len(usable)] = np.asarray(usable, dtype=np.float32)
    return padded[np.newaxis, ...]


def holistic_context():
    """Crea una instancia de MediaPipe Holistic para usar como context manager por conexión.

    Holistic mantiene recursos nativos; conviene crear una instancia por WebSocket y cerrarla al
    desconectar (ver la ruta translate). Los umbrales de 0.5 equilibran detección y estabilidad
    del tracking para uso con webcam.
    """
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
