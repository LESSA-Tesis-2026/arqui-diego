"use client";

import { useCallback, useRef } from "react";

import { frameMessage, type InferenceMode } from "@/lib/websocket";

const frameIntervalMs = 125;
const frameWidth = 640;
const jpegQuality = 0.92;

type UseFrameStreamingOptions = {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  selectedModeRef: React.RefObject<InferenceMode>;
  socketRef: React.RefObject<WebSocket | null>;
  videoRef: React.RefObject<HTMLVideoElement | null>;
};

export function useFrameStreaming({ canvasRef, selectedModeRef, socketRef, videoRef }: UseFrameStreamingOptions) {
  const videoFrameCallbackRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameSentAtRef = useRef(0);

  const stopFrameLoop = useCallback(() => {
    const video = videoRef.current;
    if (video && videoFrameCallbackRef.current !== null && "cancelVideoFrameCallback" in video) {
      video.cancelVideoFrameCallback(videoFrameCallbackRef.current);
      videoFrameCallbackRef.current = null;
    }

    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, [videoRef]);

  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const socket = socketRef.current;

    if (!video || !canvas || socket?.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth || !video.videoHeight) return;
    // Descartar fotogramas mientras el socket tiene bytes pendientes mantiene la UI receptiva
    // y evita que los fotogramas obsoletos en cola produzcan predicciones retrasadas.
    if (socket.bufferedAmount > 0) return;

    const width = frameWidth;
    const height = Math.round((video.videoHeight / video.videoWidth) * width);
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.drawImage(video, 0, 0, width, height);
    socket.send(frameMessage(canvas.toDataURL("image/jpeg", jpegQuality), selectedModeRef.current));
  }, [canvasRef, selectedModeRef, socketRef, videoRef]);

  const startFrameLoop = useCallback(() => {
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
  }, [sendFrame, socketRef, stopFrameLoop, videoRef]);

  return { startFrameLoop, stopFrameLoop };
}
