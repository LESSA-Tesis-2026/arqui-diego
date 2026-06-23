from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np

from app.core.config import Settings
from app.lessa.schemas import PredictionType, ResolvedMode


@dataclass
class HybridTranslationSession:
    """Per-WebSocket translation state.

    Each browser connection owns its own buffers so voting, generated text, and
    hold-last-reading behavior never leak between users or tabs.
    """

    settings: Settings
    word_sequence: collections.deque[np.ndarray] = field(init=False)
    word_buffer: collections.deque[str] = field(init=False)
    letter_buffer: collections.deque[str] = field(init=False)
    text: str = ""
    emitted_tokens: list[str] = field(default_factory=list)
    last_emitted_word: str = "nada"
    last_emitted_letter: str = "-"
    last_token_type: PredictionType = "none"
    nada_counter: int = 0
    dynamic_frames: int = 0
    static_frames: int = 0
    hands_detected_since: float | None = None
    mode_candidate: ResolvedMode = "none"
    mode_candidate_since: float | None = None
    last_inference_at: float = 0.0
    last_hands_seen_at: float | None = None
    last_resolved_mode: ResolvedMode = "none"

    def __post_init__(self) -> None:
        self.word_sequence = collections.deque(maxlen=self.settings.window_size)
        self.word_buffer = collections.deque(maxlen=self.settings.voting_buffer_size)
        self.letter_buffer = collections.deque(maxlen=self.settings.alphabet_voting_buffer_size)

    def reset(self) -> None:
        self.word_sequence.clear()
        self.word_buffer.clear()
        self.letter_buffer.clear()
        self.text = ""
        self.emitted_tokens.clear()
        self.last_emitted_word = "nada"
        self.last_emitted_letter = "-"
        self.last_token_type = "none"
        self.nada_counter = 0
        self.dynamic_frames = 0
        self.static_frames = 0
        self.hands_detected_since = None
        self.mode_candidate = "none"
        self.mode_candidate_since = None
        self.last_inference_at = 0.0
        self.last_hands_seen_at = None
        self.last_resolved_mode = "none"

    @property
    def sentence(self) -> list[str]:
        return self.emitted_tokens[-self.settings.max_sentence_words :]

    @property
    def display_text(self) -> str:
        return self.text.strip()

    def append_word(self, word: str) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self.text += f"{word} "
        self.emitted_tokens.append(word)
        self.last_token_type = "word"

    def append_letter(self, letter: str) -> None:
        if self.last_token_type == "word" and self.text and not self.text.endswith(" "):
            self.text += " "
        self.text += letter
        self.emitted_tokens.append(letter)
        self.last_token_type = "letter"
