import { useState, useRef, useCallback } from "react";
import { Upload, Play, Pause, Volume2, VolumeX, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface VideoUploaderProps {
  onVideoLoad: (file: File, url: string) => void;
  videoUrl: string | null;
  onClickPosition?: (x: number, y: number) => void;
  maskOverlay?: { x: number; y: number; active: boolean } | null;
  isProcessing?: boolean;
}

const VideoUploader = ({ 
  onVideoLoad, 
  videoUrl, 
  onClickPosition,
  maskOverlay,
  isProcessing 
}: VideoUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("video/")) {
      const url = URL.createObjectURL(file);
      onVideoLoad(file, url);
    }
  }, [onVideoLoad]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      onVideoLoad(file, url);
    }
  }, [onVideoLoad]);

  const handleVideoClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!onClickPosition || !containerRef.current || !videoRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    
    onClickPosition(x, y);
  }, [onClickPosition]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const formatTime = (time: number) => {
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const seekTo = (e: React.MouseEvent<HTMLDivElement>) => {
    if (videoRef.current) {
      const rect = e.currentTarget.getBoundingClientRect();
      const percent = (e.clientX - rect.left) / rect.width;
      videoRef.current.currentTime = percent * duration;
    }
  };

  if (!videoUrl) {
    return (
      <div
        className={cn(
          "relative flex flex-col items-center justify-center h-full min-h-[300px] rounded-xl border-2 border-dashed transition-all duration-300 cursor-pointer cyber-grid",
          isDragging
            ? "border-primary bg-primary/5 glow-primary"
            : "border-border/50 hover:border-primary/50 hover:bg-secondary/30"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
        <div className="flex flex-col items-center gap-4 p-8 text-center">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
            <Upload className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">
              Drop your video here
            </h3>
            <p className="text-sm text-muted-foreground mt-1">
              or click to browse • Max 10s, 480p
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="px-2 py-1 rounded bg-secondary/50">MP4</span>
            <span className="px-2 py-1 rounded bg-secondary/50">WebM</span>
            <span className="px-2 py-1 rounded bg-secondary/50">MOV</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 h-full">
      <div 
        ref={containerRef}
        className="relative flex-1 rounded-xl overflow-hidden bg-secondary/30 border border-border/50 cursor-crosshair"
        onClick={handleVideoClick}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full h-full object-contain"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
        />
        
        {/* Processing overlay */}
        {isProcessing && (
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
            <p className="text-sm font-mono text-primary">Processing with SAM2...</p>
            <div className="absolute inset-0 overflow-hidden">
              <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent animate-scan" />
            </div>
          </div>
        )}

        {/* Mask overlay */}
        {maskOverlay?.active && (
          <div
            className="absolute w-24 h-24 -translate-x-1/2 -translate-y-1/2 rounded-full mask-overlay animate-pulse-glow pointer-events-none"
            style={{
              left: `${maskOverlay.x}%`,
              top: `${maskOverlay.y}%`,
            }}
          >
            <div className="absolute inset-0 rounded-full border-4 border-primary animate-ping opacity-50" />
          </div>
        )}

        {/* Click hint */}
        {onClickPosition && !maskOverlay?.active && !isProcessing && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full glass-panel text-xs text-muted-foreground">
            Click on object to track
          </div>
        )}
      </div>

      {/* Video controls */}
      <div className="glass-panel rounded-lg p-3">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={togglePlay}
            className="shrink-0"
          >
            {isPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4" />
            )}
          </Button>

          <div className="flex-1 flex items-center gap-2">
            <span className="text-xs font-mono text-muted-foreground w-10">
              {formatTime(currentTime)}
            </span>
            <div
              className="flex-1 h-1.5 bg-secondary rounded-full cursor-pointer overflow-hidden"
              onClick={seekTo}
            >
              <div
                className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all"
                style={{ width: `${(currentTime / duration) * 100 || 0}%` }}
              />
            </div>
            <span className="text-xs font-mono text-muted-foreground w-10">
              {formatTime(duration)}
            </span>
          </div>

          <Button
            variant="ghost"
            size="icon"
            onClick={toggleMute}
            className="shrink-0"
          >
            {isMuted ? (
              <VolumeX className="w-4 h-4" />
            ) : (
              <Volume2 className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default VideoUploader;
