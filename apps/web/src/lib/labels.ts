const displayLabels: Record<string, string> = {
  buenos_dias: "buenos días",
  "como estas": "cómo estás",
  "cual es tu nombre": "cuál es tu nombre",
  adios: "adiós",
  perdon: "perdón",
  "por que": "por qué",
  si: "sí",
  talvez: "tal vez",
  cuidate: "cuídate",
  mi_nombre_es: "mi nombre es",
};

export function formatLabel(label: string) {
  return displayLabels[label] ?? label.replaceAll("_", " ");
}

export function sentenceCase(value: string) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

export function formatDisplay(label: string) {
  return sentenceCase(formatLabel(label));
}

export function lastWords(value: string, limit = 3) {
  return value.trim().split(/\s+/).filter(Boolean).slice(-limit).join(" ");
}

export function normalizeLiveStatus(value: string) {
  const match = value.match(/^(mantén la seña) (\d+(?:\.\d+)?)s$/i);
  if (!match) return value;

  const seconds = Number(match[2]);
  if (!Number.isFinite(seconds)) return value;

  const bucket = Math.max(0, Math.ceil(seconds * 2) / 2).toFixed(1);
  return `${match[1]} ${bucket}s`;
}
