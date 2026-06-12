import { WS_BASE_URL } from "@/lib/api";

export type TopPrediction = {
  label: string;
  confidence: number;
};

export type TranslationMessage = {
  type: "translation" | "reset" | "error";
  model_available?: boolean;
  prediction?: string;
  confidence?: number;
  stable_word?: string | null;
  emitted_word?: string | null;
  sentence?: string[];
  text?: string;
  top?: TopPrediction[];
  has_hands?: boolean;
  status: string;
  error?: string | null;
};

export function createTranslationSocket(): WebSocket {
  return new WebSocket(`${WS_BASE_URL}/api/v1/translate/stream`);
}

export function frameMessage(frame: string) {
  return JSON.stringify({ type: "frame", frame });
}

export function resetMessage() {
  return JSON.stringify({ type: "reset" });
}
