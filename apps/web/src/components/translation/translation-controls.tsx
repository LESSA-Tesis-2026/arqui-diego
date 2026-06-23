"use client";

import { Volume2, VolumeX } from "lucide-react";

import { Button } from "@/components/ui/button";
import { modeLabels, type InferenceMode, type ModelAvailability } from "@/components/translation/types";

type TranslationControlsProps = {
  modelAvailability: ModelAvailability;
  modelWarning: string | null;
  onSelectMode: (mode: InferenceMode) => void;
  onToggleVoice: () => void;
  selectedMode: InferenceMode;
  speechSupported: boolean;
  voiceControlLabel: string;
  voiceEnabled: boolean;
};

export function TranslationControls({
  modelAvailability,
  modelWarning,
  onSelectMode,
  onToggleVoice,
  selectedMode,
  speechSupported,
  voiceControlLabel,
  voiceEnabled,
}: TranslationControlsProps) {
  return (
    <div className="mt-8 flex flex-wrap items-center gap-3">
      <div className="inline-flex rounded-full border border-border/70 bg-card/65 p-1 shadow-lg shadow-primary/5 backdrop-blur">
        {(Object.keys(modeLabels) as InferenceMode[]).map((mode) => {
          const unavailable =
            mode === "words" ? !modelAvailability.word : mode === "alphabet" ? !modelAvailability.alphabet : false;
          const selected = selectedMode === mode;
          return (
            <Button
              key={mode}
              type="button"
              variant={selected ? "default" : "ghost"}
              size="sm"
              disabled={unavailable}
              onClick={() => onSelectMode(mode)}
              className="rounded-full px-4"
            >
              {modeLabels[mode]}
            </Button>
          );
        })}
      </div>
      <Button
        type="button"
        variant={voiceEnabled && speechSupported ? "default" : "ghost"}
        size="sm"
        disabled={!speechSupported}
        onClick={onToggleVoice}
        aria-pressed={voiceEnabled && speechSupported}
        aria-label={voiceEnabled ? "Silenciar voz de traducción" : "Activar voz de traducción"}
        className="rounded-full px-4"
      >
        {voiceEnabled && speechSupported ? <Volume2 className="size-4" /> : <VolumeX className="size-4" />}
        {voiceEnabled && speechSupported ? "Silenciar voz" : "Voz"}
      </Button>
      <span className="rounded-full bg-card/55 px-3 py-1.5 text-sm text-muted-foreground ring-1 ring-border/60">
        {voiceControlLabel}
      </span>
      {modelWarning ? (
        <span className="rounded-full bg-destructive/10 px-3 py-1.5 text-sm text-destructive ring-1 ring-destructive/15">
          {modelWarning}
        </span>
      ) : null}
    </div>
  );
}
