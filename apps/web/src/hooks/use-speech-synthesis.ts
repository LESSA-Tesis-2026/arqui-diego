"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type SpeakOptions = {
  lang?: string;
  pitch?: number;
  rate?: number;
  volume?: number;
};

const defaultSpeakOptions: Required<SpeakOptions> = {
  lang: "es-SV",
  pitch: 1,
  rate: 0.95,
  volume: 1,
};

function pickSpanishVoice(voices: SpeechSynthesisVoice[]) {
  const exactSalvadoran = voices.find((voice) => voice.lang.toLowerCase() === "es-sv");
  if (exactSalvadoran) return exactSalvadoran;

  const exactSpain = voices.find((voice) => voice.lang.toLowerCase() === "es-es");
  if (exactSpain) return exactSpain;

  return voices.find((voice) => voice.lang.toLowerCase().startsWith("es-")) ?? null;
}

export function useSpeechSynthesis() {
  const synthesisRef = useRef<SpeechSynthesis | null>(null);
  const pendingUtterancesRef = useRef(0);
  const [supported, setSupported] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [voice, setVoice] = useState<SpeechSynthesisVoice | null>(null);

  useEffect(() => {
    let synthesis: SpeechSynthesis | null = null;
    let voiceSyncTimeout: number | null = null;
    let syncVoices: (() => void) | null = null;

    const supportSyncTimeout = window.setTimeout(() => {
      synthesis = "speechSynthesis" in window ? window.speechSynthesis : null;
      setSupported(Boolean(synthesis));
      if (!synthesis) return;

      synthesisRef.current = synthesis;
      syncVoices = () => {
        if (!synthesis) return;
        setVoice(pickSpanishVoice(synthesis.getVoices()));
      };

      // Browser voices often arrive asynchronously after hydration, especially in Chromium.
      voiceSyncTimeout = window.setTimeout(syncVoices, 0);
      synthesis.addEventListener("voiceschanged", syncVoices);
    }, 0);

    return () => {
      window.clearTimeout(supportSyncTimeout);
      if (voiceSyncTimeout !== null) window.clearTimeout(voiceSyncTimeout);
      if (synthesis && syncVoices) synthesis.removeEventListener("voiceschanged", syncVoices);
      synthesis?.cancel();
      synthesisRef.current = null;
      pendingUtterancesRef.current = 0;
    };
  }, []);

  const cancel = useCallback(() => {
    const synthesis = synthesisRef.current;
    if (!synthesis) return;

    pendingUtterancesRef.current = 0;
    synthesis.cancel();
    setSpeaking(false);
  }, []);

  const speak = useCallback(
    (text: string, options?: SpeakOptions) => {
      const synthesis = synthesisRef.current;
      const trimmed = text.trim();
      if (!synthesis || !trimmed) return;

      const settings = { ...defaultSpeakOptions, ...options };
      const utterance = new SpeechSynthesisUtterance(trimmed);
      utterance.lang = voice?.lang ?? settings.lang;
      utterance.pitch = settings.pitch;
      utterance.rate = settings.rate;
      utterance.volume = settings.volume;
      utterance.voice = voice;

      pendingUtterancesRef.current += 1;
      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => {
        pendingUtterancesRef.current = Math.max(0, pendingUtterancesRef.current - 1);
        setSpeaking(pendingUtterancesRef.current > 0 || synthesis.speaking);
      };
      utterance.onerror = () => {
        pendingUtterancesRef.current = Math.max(0, pendingUtterancesRef.current - 1);
        setSpeaking(pendingUtterancesRef.current > 0 || synthesis.speaking);
      };

      synthesis.speak(utterance);
      setSpeaking(true);
    },
    [voice],
  );

  return { cancel, speak, speaking, supported, voice };
}
