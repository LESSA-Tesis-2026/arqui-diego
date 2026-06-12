import os
import re

import cv2

# Thickness in pixels for the IDLE / COUNTDOWN / RECORDING state frame in the capture preview.
STATE_BORDER_THICKNESS_PX = 14


def create_video_folders(videos_folder: str, words: list[str]):
    """Create output folders per word if they do not exist.

    Args:
        videos_folder: Root folder where per-word subfolders are stored.
        words: List of word labels used as subfolder names.
    """
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)
    for word in words:
        word_path = os.path.join(videos_folder, word)
        if not os.path.exists(word_path):
            os.makedirs(word_path)


def extract_sample_indices(word_folder: str) -> set[int]:
    """Return numeric sample indices found in `sample_<n>.avi` files.

    Args:
        word_folder: Folder containing captured videos for a word.

    Returns:
        Set of parsed integer indices from valid filenames.
    """
    indices: set[int] = set()
    pattern = re.compile(r"^sample_(\d+)\.avi$")
    for filename in os.listdir(word_folder):
        match = pattern.fullmatch(filename)
        if match:
            indices.add(int(match.group(1)))
    return indices


def build_capture_plan(word_folder: str, target_samples: int) -> tuple[list[int], list[int]]:
    """Build a gap-aware capture plan for a target sample count.

    The plan always fills missing indices in `[0, target_samples - 1]` first.

    Args:
        word_folder: Folder containing current video files.
        target_samples: Desired number of samples for the word.

    Returns:
        A tuple:
        - planned_indices: sorted missing indices to capture next
        - extra_indices: sorted indices >= target_samples
    """
    existing_indices = extract_sample_indices(word_folder)
    expected_indices = set(range(target_samples))
    planned_indices = sorted(expected_indices - existing_indices)
    extra_indices = sorted(idx for idx in existing_indices if idx >= target_samples)
    return planned_indices, extra_indices


def format_seconds(value: float) -> str:
    """Format seconds for UI using integer seconds."""
    return str(max(0, int(round(value))))


def format_seconds_decimal(value: float, decimals: int = 1) -> str:
    """Format seconds for UI using decimal precision."""
    return f"{max(0.0, value):.{decimals}f}"


def draw_state_border(image, phase: str):
    """Draw a frame border with a color that matches capture state.

    Colors:
        - IDLE: gray
        - COUNTDOWN: yellow
        - RECORDING: red
    """
    height, width = image.shape[:2]
    if phase == "RECORDING":
        color = (0, 0, 255)  # red
    elif phase == "COUNTDOWN":
        color = (0, 255, 255)  # yellow
    else:
        color = (160, 160, 160)  # gray

    cv2.rectangle(
        image,
        (0, 0),
        (width - 1, height - 1),
        color,
        STATE_BORDER_THICKNESS_PX,
    )


def render_status_overlay(
    image,
    word: str,
    sample_idx: int,
    target_samples: int,
    phase: str,
    countdown_seconds: float,
    record_duration_seconds: float,
    auto_stop_recording: bool,
    elapsed: float = 0.0,
):
    """Render status text for each capture phase.

    Args:
        image: Frame shown on screen.
        word: Current word label.
        sample_idx: Capture progress index (0-based) for current session.
        target_samples: Total captures planned in current session.
        phase: Current state-machine phase.
        countdown_seconds: Countdown length before recording starts.
        record_duration_seconds: Target duration for recording phase.
        auto_stop_recording: Whether recording is auto-stopped by duration.
        elapsed: Elapsed time in the current phase.
    """
    cv2.putText(
        image,
        f"Palabra: {word} | Video: {sample_idx + 1}/{target_samples}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    if phase == "IDLE":
        message = "Press 'r' to start capture"
        color = (0, 255, 0)
    elif phase == "COUNTDOWN":
        remaining = max(0.0, countdown_seconds - elapsed)
        message = f"Starts in {format_seconds(remaining)}s"
        color = (0, 255, 255)
    elif phase == "RECORDING":
        if auto_stop_recording:
            message = (
                f"RECORDING {format_seconds_decimal(elapsed)}/"
                f"{format_seconds_decimal(record_duration_seconds)}s"
            )
        else:
            message = f"RECORDING {format_seconds_decimal(elapsed)}s (press 'r' to stop)"
        color = (0, 0, 255)
    else:
        message = "Unknown state"
        color = (255, 255, 255)

    cv2.putText(image, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
