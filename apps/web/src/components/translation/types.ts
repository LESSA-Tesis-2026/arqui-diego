import type { InferenceMode, ResolvedMode, TopPrediction } from "@/lib/websocket";

export type CameraState = "idle" | "requesting" | "active" | "denied";
export type TranslationState = "idle" | "connecting" | "active" | "paused" | "unavailable";
export type PredictionKind = "word" | "letter" | "none";

export type PredictionHistoryItem = {
  label: string;
  confidence: number;
  kind: PredictionKind;
};

export type ModelAvailability = {
  word: boolean;
  alphabet: boolean;
};

export const modeLabels: Record<InferenceMode, string> = {
  auto: "Auto",
  words: "Palabras",
  alphabet: "Alfabeto",
};

export const resolvedModeLabels: Record<ResolvedMode, string> = {
  words: "Palabras",
  alphabet: "Alfabeto",
  none: "Buscando",
};

export type { InferenceMode, ResolvedMode, TopPrediction };
