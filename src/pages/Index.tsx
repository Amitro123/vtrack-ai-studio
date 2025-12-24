import { useState, useCallback } from "react";
import Header from "@/components/Header";
import VideoUploader from "@/components/VideoUploader";
import TabPanel from "@/components/TabPanel";
import DownloadPanel from "@/components/DownloadPanel";
import ProcessingStatus from "@/components/ProcessingStatus";
import { useToast } from "@/hooks/use-toast";

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
  const [maskOverlay, setMaskOverlay] = useState<{ x: number; y: number; active: boolean } | null>(null);
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "ai"; content: string }>>([]);
  const [results, setResults] = useState({ video: false, audio: false, masked: false });
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);

  const handleVideoLoad = useCallback((file: File, url: string) => {
    setVideoFile(file);
    setVideoUrl(url);
    setMaskOverlay(null);
    setResults({ video: false, audio: false, masked: false });
    setChatMessages([]);
    
    toast({
      title: "Video loaded",
      description: `${file.name} ready for processing`,
    });
  }, [toast]);

  const handleClickPosition = useCallback(async (x: number, y: number) => {
    if (!videoFile || isProcessing) return;

    setIsProcessing(true);
    setMaskOverlay({ x, y, active: true });

    setProcessingSteps([
      { id: "sam2", label: "SAM2 point tracking", status: "processing" },
      { id: "mask", label: "Generating mask overlay", status: "pending" },
      { id: "track", label: "Tracking across frames", status: "pending" },
    ]);

    // Simulate SAM2 processing
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "sam2" ? { ...s, status: "complete" } : s.id === "mask" ? { ...s, status: "processing" } : s
      )
    );

    await new Promise((resolve) => setTimeout(resolve, 1000));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "mask" ? { ...s, status: "complete" } : s.id === "track" ? { ...s, status: "processing" } : s
      )
    );

    await new Promise((resolve) => setTimeout(resolve, 1200));
    setProcessingSteps((prev) => prev.map((s) => ({ ...s, status: "complete" })));

    setResults((prev) => ({ ...prev, masked: true }));
    setIsProcessing(false);

    toast({
      title: "Object tracked",
      description: "SAM2 mask generated successfully",
    });
  }, [videoFile, isProcessing, toast]);

  const handleChatSubmit = useCallback(async (message: string) => {
    if (!videoFile || isProcessing) return;

    setChatMessages((prev) => [...prev, { role: "user", content: message }]);
    setIsProcessing(true);

    setProcessingSteps([
      { id: "grounding", label: "GroundingDINO text→bbox", status: "processing" },
      { id: "sam2", label: "SAM2 segmentation", status: "pending" },
      { id: "demucs", label: "Demucs audio separation", status: "pending" },
      { id: "highlight", label: "Generating highlights", status: "pending" },
    ]);

    // Simulate GroundingDINO
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "grounding" ? { ...s, status: "complete" } : s.id === "sam2" ? { ...s, status: "processing" } : s
      )
    );

    // Simulate SAM2
    await new Promise((resolve) => setTimeout(resolve, 1200));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "sam2" ? { ...s, status: "complete" } : s.id === "demucs" ? { ...s, status: "processing" } : s
      )
    );

    // Simulate Demucs
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "demucs" ? { ...s, status: "complete" } : s.id === "highlight" ? { ...s, status: "processing" } : s
      )
    );

    await new Promise((resolve) => setTimeout(resolve, 800));
    setProcessingSteps((prev) => prev.map((s) => ({ ...s, status: "complete" })));

    // Simulate AI response
    const responses: Record<string, string> = {
      drums: "🥁 Drums isolated! Audio stem extracted and video regions highlighted.",
      vocals: "🎤 Vocals extracted successfully! Audio and visual sync complete.",
      guitar: "🎸 Guitar located and isolated. Audio stem ready for download.",
      piano: "🎹 Piano identified and separated. Check the audio results!",
    };

    const matchedKey = Object.keys(responses).find((key) =>
      message.toLowerCase().includes(key)
    );

    setChatMessages((prev) => [
      ...prev,
      {
        role: "ai",
        content: matchedKey
          ? responses[matchedKey]
          : `✅ Processed "${message}". Found and isolated the matching elements.`,
      },
    ]);

    setResults((prev) => ({ ...prev, audio: true, video: true }));
    setIsProcessing(false);

    toast({
      title: "Processing complete",
      description: "Audio and video results ready",
    });
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

    setProcessingSteps([
      { id: "track", label: "SAM2 temporal tracking", status: "processing" },
      { id: "mask", label: "Generating removal mask", status: "pending" },
      { id: "inpaint", label: "ProPainter inpainting", status: "pending" },
      { id: "render", label: "Rendering final video", status: "pending" },
    ]);

    // Simulate tracking
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "track" ? { ...s, status: "complete" } : s.id === "mask" ? { ...s, status: "processing" } : s
      )
    );

    // Simulate mask generation
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "mask" ? { ...s, status: "complete" } : s.id === "inpaint" ? { ...s, status: "processing" } : s
      )
    );

    // Simulate inpainting (longer)
    await new Promise((resolve) => setTimeout(resolve, 3000));
    setProcessingSteps((prev) =>
      prev.map((s) =>
        s.id === "inpaint" ? { ...s, status: "complete" } : s.id === "render" ? { ...s, status: "processing" } : s
      )
    );

    // Simulate rendering
    await new Promise((resolve) => setTimeout(resolve, 1200));
    setProcessingSteps((prev) => prev.map((s) => ({ ...s, status: "complete" })));

    setResults((prev) => ({ ...prev, video: true }));
    setMaskOverlay(null);
    setIsProcessing(false);

    toast({
      title: "Object removed",
      description: "Inpainted video ready for download",
    });
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
            />
          </div>

          {/* Right: Tabs + Processing */}
          <div className="flex flex-col gap-4">
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
            <span className="text-accent">● CUDA Active</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
