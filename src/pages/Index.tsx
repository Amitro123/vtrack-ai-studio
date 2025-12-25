import { useState, useCallback } from "react";
import Header from "@/components/Header";
import VideoUploader from "@/components/VideoUploader";
import TabPanel from "@/components/TabPanel";
import DownloadPanel from "@/components/DownloadPanel";
import ProcessingStatus from "@/components/ProcessingStatus";
import ProcessingModeToggle from "@/components/ProcessingModeToggle";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/lib/api";

interface ProcessingStep {
  id: string;
  label: string;
  status: "pending" | "processing" | "complete" | "error";
}

const Index = () => {
  const { toast } = useToast();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"click" | "chat" | "remove">("click");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMode, setProcessingMode] = useState<"fast" | "accurate">("fast");
  const [maskOverlay, setMaskOverlay] = useState<{ x: number; y: number; active: boolean } | null>(null);
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "ai"; content: string }>>([]);
  const [results, setResults] = useState({ video: false, audio: false, masked: false });
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [resultUrls, setResultUrls] = useState<{ video?: string; audio?: string }>({});

  const handleVideoLoad = useCallback((file: File, url: string) => {
    setVideoFile(file);
    setVideoUrl(url);
    setMaskOverlay(null);
    setResults({ video: false, audio: false, masked: false });
    setChatMessages([]);
    setResultUrls({});

    toast({
      title: "Video loaded",
      description: `${file.name} ready for processing`,
    });
  }, [toast]);

  const handleClickPosition = useCallback(async (x: number, y: number) => {
    if (!videoFile || isProcessing) return;

    setIsProcessing(true);
    setMaskOverlay({ x, y, active: true });

    try {
      // Call real backend API with processing mode
      const result = await api.trackPoint(videoFile, x, y, 0, processingMode);

      // Update processing steps from backend
      setProcessingSteps(result.processing_steps);

      // Set result video URL
      const videoUrl = api.getFileUrl(result.masked_video_url);
      setResultUrls((prev) => ({ ...prev, video: videoUrl }));
      setResults((prev) => ({ ...prev, masked: true, video: true }));

      toast({
        title: "Object tracked",
        description: "SAM2 mask generated successfully",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to track object",
        variant: "destructive",
      });
      setMaskOverlay(null);
    } finally {
      setIsProcessing(false);
    }
  }, [videoFile, isProcessing, toast]);

  const handleChatSubmit = useCallback(async (message: string) => {
    if (!videoFile || isProcessing) return;

    setChatMessages((prev) => [...prev, { role: "user", content: message }]);
    setIsProcessing(true);

    try {
      // Call real backend API with processing mode
      const result = await api.textToVideo(videoFile, message, processingMode);

      // Update processing steps from backend
      setProcessingSteps(result.processing_steps);

      // Set result URLs
      const videoUrl = api.getFileUrl(result.highlighted_video_url);
      const audioUrl = api.getFileUrl(result.audio_url);
      setResultUrls({ video: videoUrl, audio: audioUrl });
      setResults((prev) => ({ ...prev, audio: true, video: true }));

      // Add AI response to chat
      setChatMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: result.ai_response,
        },
      ]);

      toast({
        title: "Processing complete",
        description: "Audio and video results ready",
      });
    } catch (error) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: `❌ Error: ${error instanceof Error ? error.message : "Processing failed"}`,
        },
      ]);

      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  }, [videoFile, isProcessing, toast]);

  const handleRemoveAction = useCallback(async () => {
    if (!videoFile || isProcessing || !maskOverlay?.active) {
      toast({
        title: "No object selected",
        description: "Click on an object in the video first",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);

    try {
      // Call real backend API with processing mode
      const result = await api.removeObject(videoFile, maskOverlay.x, maskOverlay.y, 0, processingMode);

      // Update processing steps from backend
      setProcessingSteps(result.processing_steps);

      // Set result video URL
      const videoUrl = api.getFileUrl(result.inpainted_video_url);
      setResultUrls((prev) => ({ ...prev, video: videoUrl }));
      setResults((prev) => ({ ...prev, video: true }));
      setMaskOverlay(null);

      toast({
        title: "Object removed",
        description: "Inpainted video ready for download",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to remove object",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  }, [videoFile, isProcessing, maskOverlay, toast]);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 p-4 md:p-6 max-w-7xl mx-auto w-full">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
          {/* Left: Video Player */}
          <div className="flex flex-col gap-4">
            <div className="flex-1 min-h-[400px]">
              <VideoUploader
                onVideoLoad={handleVideoLoad}
                videoUrl={videoUrl}
                onClickPosition={activeTab === "click" ? handleClickPosition : undefined}
                maskOverlay={maskOverlay}
                isProcessing={isProcessing}
              />
            </div>

            <DownloadPanel
              hasResults={results.video || results.audio || results.masked}
              results={results}
              isProcessing={isProcessing}
              videoUrl={resultUrls.video}
              audioUrl={resultUrls.audio}
            />
          </div>

          {/* Right: Tabs + Processing */}
          <div className="flex flex-col gap-4">
            {/* Processing Mode Toggle */}
            <ProcessingModeToggle
              mode={processingMode}
              onModeChange={setProcessingMode}
              disabled={isProcessing}
            />

            <div className="flex-1 min-h-[400px]">
              <TabPanel
                activeTab={activeTab}
                onTabChange={setActiveTab}
                onChatSubmit={handleChatSubmit}
                onRemoveAction={handleRemoveAction}
                isProcessing={isProcessing}
                chatMessages={chatMessages}
                hasVideo={!!videoFile}
              />
            </div>

            <ProcessingStatus
              steps={processingSteps}
              visible={isProcessing || processingSteps.some((s) => s.status === "complete")}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="glass-panel border-t border-border/50 px-6 py-3 mt-auto">
        <div className="flex items-center justify-between max-w-7xl mx-auto text-xs text-muted-foreground">
          <span className="font-mono">VTrackAI v1.0</span>
          <div className="flex items-center gap-4">
            <span>Max: 10s @ 480p</span>
            <span className="text-accent">● Backend Connected</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
