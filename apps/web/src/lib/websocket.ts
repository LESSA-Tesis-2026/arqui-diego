import { WS_BASE_URL } from "@/lib/api";

export type InferenceMode = "auto" | "words" | "alphabet";
export type ResolvedMode = "words" | "alphabet" | "none";
export type PredictionType = "word" | "letter" | "none";

export type TopPrediction = {
  label: string;
  confidence: number;
};

export type TranslationMessage = {
  type: "translation" | "reset" | "error";
  requested_mode?: InferenceMode;
  mode?: ResolvedMode;
  prediction_type?: PredictionType;
  word_available?: boolean;
  alphabet_available?: boolean;
  prediction?: string | null;
  confidence?: number;
  stable_word?: string | null;
  emitted_word?: string | null;
  emitted_token?: string | null;
  sentence?: string[];
  text?: string;
  top?: TopPrediction[];
  word_top?: TopPrediction[];
  alphabet_top?: TopPrediction[];
  has_hands?: boolean;
  motion_score?: number;
  status: string;
  error?: string | null;
};

export function createTranslationSocket(): WebSocket {
  return new WebSocket(`${WS_BASE_URL}/api/v1/translate/stream`);
}

export function frameMessage(frame: string, mode: InferenceMode) {
  return JSON.stringify({ type: "frame", frame, mode });
}

export function resetMessage() {
  return JSON.stringify({ type: "reset" });
}
