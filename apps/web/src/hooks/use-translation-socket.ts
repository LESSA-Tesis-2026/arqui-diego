"use client";

import { useCallback, useEffect, useState } from "react";

import type {
  InferenceMode,
  ModelAvailability,
  PredictionHistoryItem,
  PredictionKind,
  ResolvedMode,
} from "@/components/translation/types";
import { getModelInfo } from "@/lib/api";
import { formatDisplay, formatLabel, normalizeLiveStatus } from "@/lib/labels";
import {
  createTranslationSocket,
  resetMessage,
  type TopPrediction,
  type TranslationMessage,
} from "@/lib/websocket";
import { useSpeechSynthesis } from "@/hooks/use-speech-synthesis";

function setIfChanged<T>(setter: (value: T | ((current: T) => T)) => void, value: T) {
  setter((current: T) => (Object.is(current, value) ? current : value));
}

type UseTranslationSocketOptions = {
  selectedModeRef: React.RefObject<InferenceMode>;
  socketRef: React.RefObject<WebSocket | null>;
  startFrameLoop: () => void;
  stopFrameLoop: () => void;
};

export function useTranslationSocket({
  selectedModeRef,
  socketRef,
  startFrameLoop,
  stopFrameLoop,
}: UseTranslationSocketOptions) {
  const [translationState, setTranslationState] = useState<"idle" | "connecting" | "active" | "paused" | "unavailable">("idle");
  const [status, setStatus] = useState("Listo para traducir");
  const [text, setText] = useState("");
  const [latestSign, setLatestSign] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [hasHands, setHasHands] = useState(false);
  const [top, setTop] = useState<TopPrediction[]>([]);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [activeMode, setActiveMode] = useState<ResolvedMode>("none");
  const [predictionType, setPredictionType] = useState<PredictionKind>("none");
  const [isHoldingReading, setIsHoldingReading] = useState(false);
  const [motionScore, setMotionScore] = useState(0);
  const [modelAvailability, setModelAvailability] = useState<ModelAvailability>({
    word: true,
    alphabet: true,
  });
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const { cancel: cancelSpeech, speak, speaking, supported: speechSupported } = useSpeechSynthesis();

  const stopStreaming = useCallback(() => {
    cancelSpeech();
    stopFrameLoop();
    socketRef.current?.close();
    socketRef.current = null;
  }, [cancelSpeech, socketRef, stopFrameLoop]);

  const handleTranslationMessage = useCallback(
    (message: TranslationMessage) => {
      if (message.type === "error") {
        setIfChanged(setStatus, normalizeLiveStatus(message.status));
        return;
      }

      const nextStatus = normalizeLiveStatus(message.status);
      const nextMode = message.mode ?? "none";
      const nextPredictionType = message.prediction_type ?? "none";
      const nextMotionScore = Number((message.motion_score ?? 0).toFixed(4));
      const shouldHoldPreviousReading =
        nextPredictionType === "none" &&
        nextMode !== "none" &&
        (Boolean(message.has_hands) || nextStatus.toLowerCase().includes("manteniendo"));

      setIfChanged(setStatus, nextStatus);
      setIfChanged(setHasHands, Boolean(message.has_hands));
      setIfChanged(setActiveMode, nextMode);
      if (!shouldHoldPreviousReading) setIfChanged(setPredictionType, nextPredictionType);
      setIfChanged(setIsHoldingReading, shouldHoldPreviousReading);
      setMotionScore((current) => (Math.abs(current - nextMotionScore) < 0.0005 ? current : nextMotionScore));

      if (message.word_available !== undefined || message.alphabet_available !== undefined) {
        setModelAvailability((current) => {
          const next = {
            word: message.word_available ?? current.word,
            alphabet: message.alphabet_available ?? current.alphabet,
          };
          return current.word === next.word && current.alphabet === next.alphabet ? current : next;
        });
      }

      if (message.text !== undefined) {
        setText((current) => {
          const next = formatLabel(message.text ?? "");
          return current === next ? current : next;
        });
      } else if (message.sentence !== undefined) {
        setText((current) => {
          const next = message.sentence?.map(formatLabel).join(" ") ?? "";
          return current === next ? current : next;
        });
      }

      if (nextPredictionType === "none") {
        if (!shouldHoldPreviousReading && (nextMode === "none" || !message.has_hands)) {
          setLatestSign(null);
          setConfidence(0);
          setTop([]);
        }
      } else if (message.prediction) {
        setIfChanged(setIsHoldingReading, false);
        const nextLatestSign =
          nextPredictionType === "letter"
            ? message.prediction
            : formatDisplay(message.prediction);
        setIfChanged(setLatestSign, nextLatestSign);
        if (message.confidence !== undefined) setConfidence(message.confidence);

        const ranked =
          nextPredictionType === "letter"
            ? (message.alphabet_top ?? message.top)
            : (message.word_top ?? message.top);
        setTop(ranked ?? []);
      }

      const emitted = message.emitted_token ?? message.emitted_word;
      if (emitted) {
        // Audio is intentionally tied to stable backend emissions only; speaking raw
        // frame-level predictions would repeat unstable guesses and confuse users.
        if (voiceEnabled && speechSupported) {
          const spoken = message.prediction_type === "letter" ? emitted : formatLabel(emitted);
          speak(spoken);
        }

        setHistory((current) => [
          {
            label: message.prediction_type === "letter" ? emitted : formatDisplay(emitted),
            confidence: message.confidence ?? 0,
            kind: message.prediction_type ?? "none",
          },
          ...current,
        ].slice(0, 6));
      }
    },
    [speak, speechSupported, voiceEnabled],
  );

  const connect = useCallback(() => {
    stopStreaming();
    setTranslationState("connecting");
    setStatus("Conectando con el traductor");

    const socket = createTranslationSocket();
    socketRef.current = socket;

    socket.onopen = () => {
      setTranslationState("active");
      setStatus("Traduciendo en vivo");
      startFrameLoop();
    };

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as TranslationMessage;
      handleTranslationMessage(message);
    };

    socket.onerror = () => {
      setTranslationState("unavailable");
      setStatus("Conexión no disponible");
      stopStreaming();
    };

    socket.onclose = () => {
      stopFrameLoop();
    };
  }, [handleTranslationMessage, socketRef, startFrameLoop, stopFrameLoop, stopStreaming]);

  const pauseTranslation = useCallback(() => {
    setTranslationState("paused");
    setStatus("Traducción pausada");
    stopStreaming();
  }, [stopStreaming]);

  const clearTranslation = useCallback(() => {
    cancelSpeech();
    setText("");
    setLatestSign(null);
    setConfidence(0);
    setHasHands(false);
    setTop([]);
    setHistory([]);
    setIsHoldingReading(false);

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(resetMessage());
    }

    setStatus("Traducción reiniciada");
  }, [cancelSpeech, socketRef]);

  const toggleVoiceFeedback = useCallback(() => {
    setVoiceEnabled((current) => {
      if (current) cancelSpeech();
      return !current;
    });
  }, [cancelSpeech]);

  useEffect(() => {
    let mounted = true;

    getModelInfo()
      .then((info) => {
        if (!mounted) return;
        setModelAvailability({ word: info.word_available, alphabet: info.alphabet_available });
        setStatus(info.available ? "Modelos listos" : "Traductor en espera");
      })
      .catch(() => {
        if (!mounted) return;
        setTranslationState("unavailable");
        setStatus("No se pudo conectar con el traductor");
      });

    return () => {
      mounted = false;
      stopStreaming();
    };
  }, [stopStreaming]);

  return {
    activeMode,
    clearTranslation,
    confidence,
    connect,
    hasHands,
    history,
    isHoldingReading,
    latestSign,
    modelAvailability,
    motionScore,
    pauseTranslation,
    predictionType,
    selectedModeRef,
    speaking,
    speechSupported,
    status,
    text,
    toggleVoiceFeedback,
    top,
    translationState,
    voiceEnabled,
  };
}
