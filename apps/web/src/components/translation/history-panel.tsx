"use client";

import { AnimatePresence, motion } from "motion/react";

import type { PredictionHistoryItem } from "@/components/translation/types";

type HistoryPanelProps = {
  history: PredictionHistoryItem[];
};

export function HistoryPanel({ history }: HistoryPanelProps) {
  return (
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
  );
}
