"use client";

import { useRef, useState } from "react";
import { Pause, Waves, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { CameraPanel } from "@/components/translation/camera-panel";
import { HistoryPanel } from "@/components/translation/history-panel";
import { PredictionPanel } from "@/components/translation/prediction-panel";
import { StatusBar } from "@/components/translation/status-bar";
import { TranslationControls } from "@/components/translation/translation-controls";
import {
  resolvedModeLabels,
  type InferenceMode,
} from "@/components/translation/types";
import { useCamera } from "@/hooks/use-camera";
import { useFrameStreaming } from "@/hooks/use-frame-streaming";
import { useTranslationSocket } from "@/hooks/use-translation-socket";
import { lastWords, sentenceCase } from "@/lib/labels";

export function TranslationExperience() {
  const socketRef = useRef<WebSocket | null>(null);
  const selectedModeRef = useRef<InferenceMode>("auto");
  const [selectedMode, setSelectedMode] = useState<InferenceMode>("auto");
  const { activateCamera, cameraState, canvasRef, videoRef } = useCamera();
  const { startFrameLoop, stopFrameLoop } = useFrameStreaming({
    canvasRef,
    selectedModeRef,
    socketRef,
    videoRef,
  });
  const translation = useTranslationSocket({
    selectedModeRef,
    socketRef,
    startFrameLoop,
    stopFrameLoop,
  });

  async function startTranslation() {
    if (cameraState !== "active") {
      const activated = await activateCamera();
      if (!activated) return;
    }

    translation.connect();
  }

  function selectMode(mode: InferenceMode) {
    selectedModeRef.current = mode;
    setSelectedMode(mode);
  }

  const isActive = translation.translationState === "active" || translation.translationState === "connecting";
  const canUseSelectedMode =
    selectedMode === "auto"
      ? translation.modelAvailability.word || translation.modelAvailability.alphabet
      : selectedMode === "words"
        ? translation.modelAvailability.word
        : translation.modelAvailability.alphabet;
  const modelWarning =
    !translation.modelAvailability.word && !translation.modelAvailability.alphabet
      ? "No hay modelos disponibles"
      : !translation.modelAvailability.word
        ? "Modelo de palabras no disponible"
        : !translation.modelAvailability.alphabet
          ? "Modelo de alfabeto no disponible"
          : null;
  const voiceControlLabel = !translation.speechSupported
    ? "Voz no disponible"
    : translation.voiceEnabled
      ? translation.speaking
        ? "Hablando"
        : "Voz activa"
      : "Voz silenciada";
  const displayText = translation.text ? sentenceCase(lastWords(translation.text, 3)) : "Esperando";
  const primaryActionLabel =
    cameraState !== "active"
      ? cameraState === "requesting"
        ? "Preparando cámara"
        : "Activar cámara"
      : translation.translationState === "active"
        ? "Pausar"
        : "Iniciar traducción";
  const primaryAction = cameraState !== "active"
    ? startTranslation
    : translation.translationState === "active"
      ? translation.pauseTranslation
      : startTranslation;

  return (
    <main className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_58%_at_28%_48%,var(--card)_0%,transparent_62%),radial-gradient(ellipse_46%_54%_at_78%_20%,var(--accent)_0%,transparent_64%),radial-gradient(ellipse_42%_34%_at_82%_88%,var(--muted)_0%,transparent_68%),linear-gradient(135deg,var(--background),var(--secondary))] opacity-90" />
      <div className="absolute left-[8%] top-[16%] h-px w-[44rem] rotate-[-18deg] bg-gradient-to-r from-transparent via-border/70 to-transparent" />
      <div className="absolute bottom-[-18rem] right-[-10rem] h-[42rem] w-[42rem] rounded-full bg-accent/20 blur-3xl" />

      <section className="relative z-10 mx-auto grid min-h-screen w-full max-w-[1500px] px-5 py-6 sm:px-8 lg:px-12">
        <div className="grid items-center gap-8 py-8 lg:grid-cols-[minmax(320px,0.82fr)_minmax(520px,1.18fr)] lg:gap-14 lg:py-0">
          <CameraPanel cameraState={cameraState} canvasRef={canvasRef} videoRef={videoRef} />

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
                  className={`max-w-3xl text-balance font-medium leading-[0.98] tracking-[-0.065em] ${translation.text ? "text-5xl text-foreground sm:text-6xl lg:text-8xl" : "text-4xl text-muted-foreground/70 sm:text-5xl lg:text-7xl"}`}
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
                  {isActive && !translation.hasHands ? (
                    <motion.span
                      className="size-2 rounded-full bg-accent"
                      animate={{ scale: [1, 1.7, 1], opacity: [0.9, 0.35, 0.9] }}
                      transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
                    />
                  ) : translation.hasHands ? (
                    <Waves className="size-3.5 text-accent-foreground" />
                  ) : (
                    <Pause className="size-3.5" />
                  )}
                  {translation.hasHands ? "Movimiento detectado" : isActive ? "Buscando movimiento" : "Pausado"}
                </span>
                <span className="inline-flex items-center gap-2 rounded-full bg-card/55 px-3 py-1.5 ring-1 ring-border/60">
                  Modo activo: {resolvedModeLabels[translation.activeMode]}
                </span>
                {isActive ? (
                  <span className="inline-flex items-center gap-2 rounded-full bg-card/55 px-3 py-1.5 ring-1 ring-border/60">
                    Movimiento {translation.motionScore.toFixed(4)} / umbral 0.010
                  </span>
                ) : null}
              </motion.div>
            </div>

            {translation.translationState === "unavailable" || cameraState === "denied" ? (
              <div className="mt-8 flex items-center justify-center gap-3 rounded-full bg-destructive/10 px-4 py-3 text-sm text-destructive ring-1 ring-destructive/15 lg:justify-start">
                <X className="size-4" />
                {cameraState === "denied"
                  ? "Permite el acceso a la cámara para traducir."
                  : "El traductor no está disponible en este momento."}
              </div>
            ) : null}

            <TranslationControls
              modelAvailability={translation.modelAvailability}
              modelWarning={modelWarning}
              onSelectMode={selectMode}
              onToggleVoice={translation.toggleVoiceFeedback}
              selectedMode={selectedMode}
              speechSupported={translation.speechSupported}
              voiceControlLabel={voiceControlLabel}
              voiceEnabled={translation.voiceEnabled}
            />

            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, delay: 0.18, ease: [0.22, 1, 0.36, 1] }}
              className="mt-14 grid gap-14 lg:grid-cols-[minmax(280px,0.92fr)_minmax(220px,0.58fr)]"
            >
              <PredictionPanel
                confidence={translation.confidence}
                isHoldingReading={translation.isHoldingReading}
                latestSign={translation.latestSign}
                predictionType={translation.predictionType}
                top={translation.top}
              />
              <HistoryPanel history={translation.history} />
            </motion.div>
          </motion.div>
        </div>
      </section>

      <StatusBar
        canUseSelectedMode={cameraState !== "active" || canUseSelectedMode}
        cameraIsRequesting={cameraState === "requesting"}
        isConnecting={translation.translationState === "connecting"}
        onClear={translation.clearTranslation}
        onPrimaryAction={primaryAction}
        primaryActionLabel={primaryActionLabel}
        status={translation.status}
        translationIsActive={translation.translationState === "active"}
      />
    </main>
  );
}
