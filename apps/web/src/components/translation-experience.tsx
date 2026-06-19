"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw, Volume2, VolumeX, Waves, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { Button } from "@/components/ui/button";
import { getModelInfo } from "@/lib/api";
import { useSpeechSynthesis } from "@/lib/use-speech-synthesis";
import {
  createTranslationSocket,
  frameMessage,
  resetMessage,
  type InferenceMode,
  type ResolvedMode,
  type TopPrediction,
  type TranslationMessage,
} from "@/lib/websocket";

type CameraState = "idle" | "requesting" | "active" | "denied";
type TranslationState = "idle" | "connecting" | "active" | "paused" | "unavailable";
type PredictionHistoryItem = {
  label: string;
  confidence: number;
  kind: "word" | "letter" | "none";
};

type ModelAvailability = {
  word: boolean;
  alphabet: boolean;
};

const frameIntervalMs = 125;
const frameWidth = 640;
const jpegQuality = 0.92;

const modeLabels: Record<InferenceMode, string> = {
  auto: "Auto",
  words: "Palabras",
  alphabet: "Alfabeto",
};

const resolvedModeLabels: Record<ResolvedMode, string> = {
  words: "Palabras",
  alphabet: "Alfabeto",
  none: "Buscando",
};

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

function formatLabel(label: string) {
  return displayLabels[label] ?? label.replaceAll("_", " ");
}

function formatDisplay(label: string) {
  return sentenceCase(formatLabel(label));
}

function lastWords(value: string, limit = 3) {
  return value.trim().split(/\s+/).filter(Boolean).slice(-limit).join(" ");
}

function sentenceCase(value: string) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

function normalizeLiveStatus(value: string) {
  const match = value.match(/^(mantén la seña) (\d+(?:\.\d+)?)s$/i);
  if (!match) return value;

  const seconds = Number(match[2]);
  if (!Number.isFinite(seconds)) return value;

  const bucket = Math.max(0, Math.ceil(seconds * 2) / 2).toFixed(1);
  return `${match[1]} ${bucket}s`;
}

function setIfChanged<T>(setter: (value: T | ((current: T) => T)) => void, value: T) {
  setter((current: T) => (Object.is(current, value) ? current : value));
}

export function TranslationExperience() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const videoFrameCallbackRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameSentAtRef = useRef(0);
  const selectedModeRef = useRef<InferenceMode>("auto");

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [translationState, setTranslationState] = useState<TranslationState>("idle");
  const [status, setStatus] = useState("Listo para traducir");
  const [text, setText] = useState("");
  const [latestSign, setLatestSign] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [hasHands, setHasHands] = useState(false);
  const [top, setTop] = useState<TopPrediction[]>([]);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [selectedMode, setSelectedMode] = useState<InferenceMode>("auto");
  const [activeMode, setActiveMode] = useState<ResolvedMode>("none");
  const [predictionType, setPredictionType] = useState<"word" | "letter" | "none">("none");
  const [isHoldingReading, setIsHoldingReading] = useState(false);
  const [motionScore, setMotionScore] = useState(0);
  const [modelAvailability, setModelAvailability] = useState<ModelAvailability>({
    word: true,
    alphabet: true,
  });
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const { cancel: cancelSpeech, speak, speaking, supported: speechSupported } = useSpeechSynthesis();

  function stopFrameLoop() {
    const video = videoRef.current;
    if (video && videoFrameCallbackRef.current !== null && "cancelVideoFrameCallback" in video) {
      video.cancelVideoFrameCallback(videoFrameCallbackRef.current);
      videoFrameCallbackRef.current = null;
    }

    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }

  async function activateCamera() {
    setCameraState("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraState("active");
      setStatus("Cámara activa");
    } catch {
      setCameraState("denied");
      setStatus("Activa la cámara para iniciar la traducción");
    }
  }

  function startTranslation() {
    if (cameraState !== "active") {
      void activateCamera();
      return;
    }

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
  }

  function pauseTranslation() {
    setTranslationState("paused");
    setStatus("Traducción pausada");
    stopStreaming();
  }

  function clearTranslation() {
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
  }

  function stopStreaming() {
    cancelSpeech();
    stopFrameLoop();
    socketRef.current?.close();
    socketRef.current = null;
  }

  function startFrameLoop() {
    stopFrameLoop();
    lastFrameSentAtRef.current = 0;
    const video = videoRef.current;

    if (video && "requestVideoFrameCallback" in video) {
      const tick: VideoFrameRequestCallback = (_now, metadata) => {
        const timestamp = metadata.mediaTime * 1000 || performance.now();
        if (timestamp - lastFrameSentAtRef.current >= frameIntervalMs) {
          lastFrameSentAtRef.current = timestamp;
          sendFrame();
        }
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          videoFrameCallbackRef.current = video.requestVideoFrameCallback(tick);
        }
      };

      videoFrameCallbackRef.current = video.requestVideoFrameCallback(tick);
      return;
    }

    const fallbackTick = (timestamp: number) => {
      if (timestamp - lastFrameSentAtRef.current >= frameIntervalMs) {
        lastFrameSentAtRef.current = timestamp;
        sendFrame();
      }

      if (socketRef.current?.readyState === WebSocket.OPEN) {
        animationFrameRef.current = window.requestAnimationFrame(fallbackTick);
      }
    };

    animationFrameRef.current = window.requestAnimationFrame(fallbackTick);
  }

  function selectMode(mode: InferenceMode) {
    selectedModeRef.current = mode;
    setSelectedMode(mode);
  }

  function toggleVoiceFeedback() {
    setVoiceEnabled((current) => {
      if (current) cancelSpeech();
      return !current;
    });
  }

  function sendFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const socket = socketRef.current;

    if (!video || !canvas || socket?.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth || !video.videoHeight) return;
    if (socket.bufferedAmount > 0) return;

    const width = frameWidth;
    const height = Math.round((video.videoHeight / video.videoWidth) * width);
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.drawImage(video, 0, 0, width, height);
    socket.send(frameMessage(canvas.toDataURL("image/jpeg", jpegQuality), selectedModeRef.current));
  }

  function handleTranslationMessage(message: TranslationMessage) {
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

    if (message.text !== undefined) setText((current) => {
      const next = formatLabel(message.text ?? "");
      return current === next ? current : next;
    });
    else if (message.sentence !== undefined) setText((current) => {
      const next = message.sentence?.map(formatLabel).join(" ") ?? "";
      return current === next ? current : next;
    });

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
      // Audio is tied to emitted tokens only, so unstable frame-level predictions never speak.
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
  }

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
      cancelSpeech();
      stopFrameLoop();
      socketRef.current?.close();
      socketRef.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [cancelSpeech]);

  const isActive = translationState === "active" || translationState === "connecting";
  const canUseSelectedMode =
    selectedMode === "auto"
      ? modelAvailability.word || modelAvailability.alphabet
      : selectedMode === "words"
        ? modelAvailability.word
        : modelAvailability.alphabet;
  const modelWarning =
    !modelAvailability.word && !modelAvailability.alphabet
      ? "No hay modelos disponibles"
      : !modelAvailability.word
        ? "Modelo de palabras no disponible"
        : !modelAvailability.alphabet
          ? "Modelo de alfabeto no disponible"
          : null;
  const voiceControlLabel = !speechSupported
    ? "Voz no disponible"
    : voiceEnabled
      ? speaking
        ? "Hablando"
        : "Voz activa"
      : "Voz silenciada";
  const displayText = text ? sentenceCase(lastWords(text, 3)) : "Esperando";
  const primaryActionLabel =
    cameraState !== "active"
      ? cameraState === "requesting"
        ? "Preparando cámara"
        : "Activar cámara"
      : translationState === "active"
        ? "Pausar"
        : "Iniciar traducción";
  const primaryAction = cameraState !== "active" ? activateCamera : translationState === "active" ? pauseTranslation : startTranslation;

  return (
    <main className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_58%_at_28%_48%,var(--card)_0%,transparent_62%),radial-gradient(ellipse_46%_54%_at_78%_20%,var(--accent)_0%,transparent_64%),radial-gradient(ellipse_42%_34%_at_82%_88%,var(--muted)_0%,transparent_68%),linear-gradient(135deg,var(--background),var(--secondary))] opacity-90" />
      <div className="absolute left-[8%] top-[16%] h-px w-[44rem] rotate-[-18deg] bg-gradient-to-r from-transparent via-border/70 to-transparent" />
      <div className="absolute bottom-[-18rem] right-[-10rem] h-[42rem] w-[42rem] rounded-full bg-accent/20 blur-3xl" />

      <section className="relative z-10 mx-auto grid min-h-screen w-full max-w-[1500px] px-5 py-6 sm:px-8 lg:px-12">
        <div className="grid items-center gap-8 py-8 lg:grid-cols-[minmax(320px,0.82fr)_minmax(520px,1.18fr)] lg:gap-14 lg:py-0">
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            className="relative mx-auto aspect-[4/5] w-full max-w-[500px] overflow-hidden rounded-[3.5rem] bg-primary shadow-2xl shadow-primary/20 ring-1 ring-border/70 sm:aspect-[4/3] lg:aspect-[4/5]"
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="h-full w-full scale-x-[-1] object-cover opacity-90"
            />
            <canvas ref={canvasRef} className="hidden" />

            <AnimatePresence>
              {cameraState !== "active" ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.35 }}
                  className="absolute inset-0 grid place-items-center overflow-hidden bg-primary p-8 text-center text-primary-foreground"
                >
                  <div className="absolute inset-0 bg-[radial-gradient(ellipse_72%_42%_at_50%_36%,var(--chart-5)_0%,transparent_62%),linear-gradient(180deg,transparent_0%,var(--primary)_78%)] opacity-55" />
                  <div className="absolute inset-x-12 top-16 h-px bg-gradient-to-r from-transparent via-primary-foreground/20 to-transparent" />
                  <div className="absolute inset-x-16 bottom-16 h-px bg-gradient-to-r from-transparent via-primary-foreground/10 to-transparent" />
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 8 }}
                  transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                  className="max-w-xs space-y-6"
                >
                  <div className="mx-auto h-px w-24 bg-gradient-to-r from-transparent via-accent to-transparent" />
                  <div>
                    <p className="text-2xl font-medium tracking-[-0.04em]">Activa tu cámara</p>
                    <p className="mt-2 text-sm leading-5 text-primary-foreground/65">
                      Mantente dentro del encuadre y comienza cuando estés listo.
                    </p>
                  </div>
                </motion.div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.75, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
            className="relative mx-auto flex min-h-[520px] w-full max-w-5xl flex-col justify-center py-6 lg:min-h-[720px] lg:py-0"
          >
            <motion.div
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 0.65, delay: 0.28, ease: [0.22, 1, 0.36, 1] }}
              className="absolute left-0 top-16 hidden h-px w-28 origin-left bg-gradient-to-r from-primary/30 to-transparent lg:block"
            />

            <div className="relative">
              <AnimatePresence mode="popLayout">
                <motion.p
                  key={displayText}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                  className={`max-w-3xl text-balance font-medium leading-[0.98] tracking-[-0.065em] ${text ? "text-5xl text-foreground sm:text-6xl lg:text-8xl" : "text-4xl text-muted-foreground/70 sm:text-5xl lg:text-7xl"}`}
                >
                  {displayText}
                </motion.p>
              </AnimatePresence>

              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.35 }}
                className="mt-8 flex flex-wrap items-center gap-3 text-sm text-muted-foreground"
              >
                <span className="inline-flex items-center gap-2 rounded-full bg-card/55 px-3 py-1.5 ring-1 ring-border/60">
                  {isActive && !hasHands ? (
                    <motion.span
                      className="size-2 rounded-full bg-accent"
                      animate={{ scale: [1, 1.7, 1], opacity: [0.9, 0.35, 0.9] }}
                      transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                    />
                  ) : hasHands ? (
                    <Waves className="size-3.5 text-accent-foreground" />
                  ) : (
                    <Pause className="size-3.5" />
                  )}
                  {hasHands ? "Movimiento detectado" : isActive ? "Buscando movimiento" : "Pausado"}
                </span>
                <span className="inline-flex items-center gap-2 rounded-full bg-card/55 px-3 py-1.5 ring-1 ring-border/60">
                  Modo activo: {resolvedModeLabels[activeMode]}
                </span>
                {isActive ? (
                  <span className="inline-flex items-center gap-2 rounded-full bg-card/55 px-3 py-1.5 ring-1 ring-border/60">
                    Movimiento {motionScore.toFixed(4)} / umbral 0.010
                  </span>
                ) : null}
              </motion.div>
            </div>

            {translationState === "unavailable" || cameraState === "denied" ? (
              <div className="mt-8 flex items-center justify-center gap-3 rounded-full bg-destructive/10 px-4 py-3 text-sm text-destructive ring-1 ring-destructive/15 lg:justify-start">
                <X className="size-4" />
                {cameraState === "denied"
                  ? "Permite el acceso a la cámara para traducir."
                  : "El traductor no está disponible en este momento."}
              </div>
            ) : null}

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
                      onClick={() => selectMode(mode)}
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
                onClick={toggleVoiceFeedback}
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
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, delay: 0.18, ease: [0.22, 1, 0.36, 1] }}
              className="mt-14 grid gap-14 lg:grid-cols-[minmax(280px,0.92fr)_minmax(220px,0.58fr)]"
            >
              <section className="max-w-xl space-y-5">
                <div className="flex items-end justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.26em] text-muted-foreground">Lectura · {isHoldingReading ? "Retenida" : predictionType === "letter" ? "Letra" : predictionType === "word" ? "Palabra" : "Espera"}</p>
                    <p className="mt-2 text-2xl font-medium tracking-[-0.05em] text-foreground">
                    {latestSign ?? "Sin seña estable"}
                    </p>
                  </div>
                  <p className="text-5xl font-medium tracking-[-0.075em] text-foreground">
                    {Math.round(confidence * 100)}%
                  </p>
                </div>

                {top.length ? (
                  <div className="space-y-3">
                    {top.map((item, index) => {
                      const value = Math.round(item.confidence * 100);
                      return (
                        <div key={item.label} className="space-y-2 text-sm text-muted-foreground">
                          <div className="flex items-center justify-between gap-4">
                            <span className="truncate">{predictionType === "letter" ? item.label : formatDisplay(item.label)}</span>
                            <span className="tabular-nums">{value}%</span>
                          </div>
                          <div className="h-2 overflow-hidden rounded-full bg-muted">
                            <motion.div
                              className="h-full rounded-full bg-gradient-to-r from-chart-1 via-chart-2 to-chart-3"
                              initial={false}
                              animate={{ width: `${value}%`, opacity: index === 0 ? 1 : 0.32 }}
                              transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="space-y-3 text-sm leading-5 text-muted-foreground">
                    <p>Las probabilidades aparecerán cuando el traductor reciba movimiento.</p>
                    <div className="space-y-2 opacity-70">
                      <div className="h-2 w-full rounded-full bg-muted" />
                      <div className="h-2 w-2/3 rounded-full bg-muted" />
                      <div className="h-2 w-1/3 rounded-full bg-muted" />
                    </div>
                  </div>
                )}
              </section>

              <section className="space-y-5">
                <p className="text-xs uppercase tracking-[0.26em] text-muted-foreground">Historial</p>
                {history.length ? (
                  <div className="space-y-2">
                    <AnimatePresence initial={false}>
                      {history.map((item, index) => (
                        <motion.div
                          key={`${item.label}-${index}`}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -8 }}
                          transition={{ duration: 0.28 }}
                          className="flex items-center justify-between gap-4 border-b border-border/60 py-2 text-sm"
                        >
                          <span className="truncate text-foreground">{item.label}</span>
                          <span className="tabular-nums text-muted-foreground">{Math.round(item.confidence * 100)}%</span>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>
                ) : (
                  <p className="max-w-64 text-sm leading-5 text-muted-foreground">
                    Las últimas señas confirmadas aparecerán aquí.
                  </p>
                )}
              </section>
            </motion.div>
          </motion.div>
        </div>
      </section>

      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.55, delay: 0.22, ease: [0.22, 1, 0.36, 1] }}
        className="fixed inset-x-0 bottom-5 z-20 px-4 sm:bottom-8"
      >
        <div className="mx-auto flex max-w-xl items-center justify-between gap-3 rounded-full border border-border/70 bg-card/85 p-2 shadow-2xl shadow-primary/15 backdrop-blur-xl">
          <p className="min-w-0 truncate pl-4 text-sm text-muted-foreground">{sentenceCase(status)}</p>
          <div className="flex shrink-0 gap-2">
            <Button
              onClick={primaryAction}
              disabled={cameraState === "requesting" || translationState === "connecting" || (cameraState === "active" && !canUseSelectedMode)}
              className="rounded-full px-5 shadow-none"
            >
              {translationState === "active" ? <Pause className="size-4" /> : <Play className="size-4" />}
              <span className="hidden sm:inline">{primaryActionLabel}</span>
            </Button>
            <Button
              onClick={clearTranslation}
              variant="ghost"
              size="icon"
              className="rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Reiniciar traducción"
            >
              <RotateCcw className="size-4" />
            </Button>
          </div>
        </div>
      </motion.div>
    </main>
  );
}
