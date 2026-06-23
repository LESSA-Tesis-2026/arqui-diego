export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

export type RuntimeModelInfo = {
  available: boolean;
  error: string | null;
  labels: string[];
  model_path: string;
  input_shape: Array<number | null> | null;
  output_shape: Array<number | null> | null;
};

export type ModelInfo = {
  available: boolean;
  word_available: boolean;
  alphabet_available: boolean;
  word: RuntimeModelInfo;
  alphabet: RuntimeModelInfo;
  sequence_length: number;
  window_size: number;
  base_feature_length: number;
  feature_length: number;
  use_temporal_features: boolean;
  temporal_delta_order: number;
  word_confidence_threshold: number;
  alphabet_confidence_threshold: number;
  hybrid_motion_threshold: number;
};

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/api/v1/model/info`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error("No se pudo conectar con el traductor.");
  }

  return response.json();
}
