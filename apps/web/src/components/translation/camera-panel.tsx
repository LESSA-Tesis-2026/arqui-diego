"use client";

import { AnimatePresence, motion } from "motion/react";

import type { CameraState } from "@/components/translation/types";

type CameraPanelProps = {
  cameraState: CameraState;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  videoRef: React.RefObject<HTMLVideoElement | null>;
};

export function CameraPanel({ cameraState, canvasRef, videoRef }: CameraPanelProps) {
  return (
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
  );
}
