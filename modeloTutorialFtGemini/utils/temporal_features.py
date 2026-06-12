"""Temporal feature helpers for sign-sequence pipelines.

This module centralizes optional temporal-feature logic so it can be enabled
or disabled from configuration without changing extraction/training code.
"""

from __future__ import annotations

import numpy as np


def get_feature_length(base_length: int, use_temporal_features: bool) -> int:
    """Return final per-frame feature length.

    Args:
        base_length: Length of position-only feature vector.
        use_temporal_features: Whether to append first-order deltas.

    Returns:
        Final feature length.
    """
    return base_length * 2 if use_temporal_features else base_length


def add_temporal_features_to_sequence(
    position_sequence: np.ndarray, use_temporal_features: bool
) -> np.ndarray:
    """Transform a full sequence into final training/inference features.

    If temporal features are enabled, concatenates position and delta per frame:
    `[pos_t, pos_t - pos_(t-1)]`, with a zero delta at the first frame.

    Args:
        position_sequence: Array of shape `(T, D)` with position-only features.
        use_temporal_features: Whether to append first-order deltas.

    Returns:
        Array of shape `(T, D)` if disabled, otherwise `(T, 2D)`.
    """
    if position_sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {position_sequence.shape}.")

    if not use_temporal_features:
        return position_sequence

    delta = np.diff(position_sequence, axis=0, prepend=position_sequence[:1])
    return np.concatenate([position_sequence, delta], axis=1)


def build_frame_features(
    current_position: np.ndarray,
    prev_position: np.ndarray | None,
    use_temporal_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one frame feature vector from position (and optional delta).

    Args:
        current_position: Position-only vector for current frame.
        prev_position: Previous frame position vector, or `None` for first frame.
        use_temporal_features: Whether to append first-order delta.

    Returns:
        Tuple `(feature_vector, new_prev_position)`.
    """
    if not use_temporal_features:
        return current_position, current_position

    if prev_position is None:
        delta = np.zeros_like(current_position)
    else:
        delta = current_position - prev_position

    return np.concatenate([current_position, delta], axis=0), current_position

