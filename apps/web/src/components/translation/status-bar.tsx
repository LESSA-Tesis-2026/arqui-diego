"use client";

import { Pause, Play, RotateCcw } from "lucide-react";
import { motion } from "motion/react";

import { Button } from "@/components/ui/button";
import { sentenceCase } from "@/lib/labels";

type StatusBarProps = {
  canUseSelectedMode: boolean;
  cameraIsRequesting: boolean;
  isConnecting: boolean;
  onClear: () => void;
  onPrimaryAction: () => void;
  primaryActionLabel: string;
  status: string;
  translationIsActive: boolean;
};

export function StatusBar({
  canUseSelectedMode,
  cameraIsRequesting,
  isConnecting,
  onClear,
  onPrimaryAction,
  primaryActionLabel,
  status,
  translationIsActive,
}: StatusBarProps) {
  return (
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
            onClick={onPrimaryAction}
            disabled={cameraIsRequesting || isConnecting || !canUseSelectedMode}
            className="rounded-full px-5 shadow-none"
          >
            {translationIsActive ? <Pause className="size-4" /> : <Play className="size-4" />}
            <span className="hidden sm:inline">{primaryActionLabel}</span>
          </Button>
          <Button
            onClick={onClear}
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
  );
}
