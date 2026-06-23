"use client";

import { motion } from "motion/react";

import { formatDisplay } from "@/lib/labels";
import type { PredictionKind, TopPrediction } from "@/components/translation/types";

type PredictionPanelProps = {
  confidence: number;
  isHoldingReading: boolean;
  latestSign: string | null;
  predictionType: PredictionKind;
  top: TopPrediction[];
};

export function PredictionPanel({ confidence, isHoldingReading, latestSign, predictionType, top }: PredictionPanelProps) {
  return (
    <section className="max-w-xl space-y-5">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.26em] text-muted-foreground">
            Lectura · {isHoldingReading ? "Retenida" : predictionType === "letter" ? "Letra" : predictionType === "word" ? "Palabra" : "Espera"}
          </p>
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
  );
}
