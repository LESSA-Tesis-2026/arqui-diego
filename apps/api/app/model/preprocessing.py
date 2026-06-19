from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


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
    if "," in frame_data:
        frame_data = frame_data.split(",", 1)[1]

    raw = base64.b64decode(frame_data)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image frame payload")
    return frame


def mediapipe_detection(image: np.ndarray, holistic) -> object:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    return holistic.process(image_rgb)


def extract_hybrid_keypoints(results: object, base_feature_length: int) -> FrameExtraction:
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
    sequence = np.asarray(position_sequence, dtype=np.float32)

    if not use_temporal_features:
        return list(sequence)

    delta = np.vstack([sequence[0:1, :], np.diff(sequence, axis=0)]).astype(np.float32)
    features = [sequence, delta]

    if temporal_delta_order >= 2:
        acceleration = np.vstack([delta[0:1, :], np.diff(delta, axis=0)]).astype(np.float32)
        features.append(acceleration)

    return list(np.concatenate(features, axis=-1).astype(np.float32))


def motion_score(position_sequence: list[np.ndarray], hand_feature_start: int = 180) -> float:
    if len(position_sequence) < 2:
        return 0.0

    sequence = np.asarray(position_sequence, dtype=np.float32)
    delta = np.vstack([sequence[0:1, :], np.diff(sequence, axis=0)]).astype(np.float32)
    recent_delta = delta[-5:, hand_feature_start:]
    return float(np.mean(np.abs(recent_delta))) if recent_delta.size else 0.0


def pad_sequence(sequence: list[np.ndarray], sequence_length: int, feature_length: int) -> np.ndarray:
    padded = np.zeros((sequence_length, feature_length), dtype=np.float32)
    usable = sequence[:sequence_length]
    if usable:
        padded[: len(usable)] = np.asarray(usable, dtype=np.float32)
    return padded[np.newaxis, ...]


def holistic_context():
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
