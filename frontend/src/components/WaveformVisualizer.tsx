
import React, { useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

interface WaveformVisualizerProps {
  playing: boolean;
  color: string;
}

const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  playing,
  color = "blue",
}) => {
  const barCount = 64;
  const containerRef = useRef<HTMLDivElement>(null);

  // Get the appropriate color class based on the selected color
  const getColorClass = (color: string) => {
    const colorMap: Record<string, string> = {
      blue: "bg-waveform-blue",
      purple: "bg-waveform-purple",
      pink: "bg-waveform-pink",
      orange: "bg-waveform-orange",
      green: "bg-waveform-green",
      red: "bg-waveform-red",
    };
    
    return colorMap[color] || "bg-waveform-blue";
  };

  // Simulate random waveform heights for visualization
  useEffect(() => {
    if (!containerRef.current || !playing) return;

    const bars = Array.from(containerRef.current.children) as HTMLElement[];
    
    const updateBars = () => {
      bars.forEach((bar) => {
        if (playing) {
          const height = Math.random() * 100;
          bar.style.height = `${height}%`;
        } else {
          bar.style.height = "20%";
        }
      });
    };

    const intervalId = setInterval(updateBars, 100);
    
    return () => clearInterval(intervalId);
  }, [playing]);

  return (
    <div 
      ref={containerRef}
      className="w-full h-40 flex items-center justify-center gap-[2px] px-4 overflow-hidden rounded-lg bg-background/50 backdrop-blur-sm border border-border/30 transition-all duration-300"
    >
      {Array.from({ length: barCount }).map((_, index) => (
        <div
          key={index}
          className={cn(
            "h-1/5 w-1 rounded-full transition-all duration-100",
            getColorClass(color),
            playing ? "opacity-100" : "opacity-70"
          )}
          style={{
            animationDelay: playing ? `${index * (0.8 / barCount)}s` : "0s",
          }}
        />
      ))}
    </div>
  );
};

export default WaveformVisualizer;
