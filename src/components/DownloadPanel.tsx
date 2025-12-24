import { Download, Video, Music, FileVideo, Check, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface DownloadPanelProps {
  hasResults: boolean;
  results: {
    video?: boolean;
    audio?: boolean;
    masked?: boolean;
  };
  isProcessing: boolean;
}

const DownloadPanel = ({ hasResults, results, isProcessing }: DownloadPanelProps) => {
  const downloadItems = [
    {
      id: "video",
      label: "Processed Video",
      icon: FileVideo,
      available: results.video,
      color: "text-primary",
    },
    {
      id: "audio",
      label: "Isolated Audio",
      icon: Music,
      available: results.audio,
      color: "text-accent",
    },
    {
      id: "masked",
      label: "Masked Video",
      icon: Video,
      available: results.masked,
      color: "text-primary",
    },
  ];

  const handleDownload = (type: string) => {
    // Mock download - in production, this would fetch from the API
    console.log(`Downloading ${type}...`);
  };

  if (!hasResults && !isProcessing) {
    return (
      <div className="glass-panel rounded-xl p-4">
        <div className="flex items-center gap-3 text-muted-foreground">
          <Download className="w-5 h-5" />
          <div>
            <p className="text-sm font-medium">No results yet</p>
            <p className="text-xs">Process a video to download results</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Download className="w-4 h-4" />
          Download Results
        </h3>
        {isProcessing && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="w-3 h-3 animate-spin" />
            Processing...
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        {downloadItems.map((item) => (
          <Button
            key={item.id}
            variant="outline"
            size="sm"
            onClick={() => handleDownload(item.id)}
            disabled={!item.available}
            className={cn(
              "flex items-center gap-2 h-auto py-2.5 justify-start",
              item.available && "hover:border-primary/50"
            )}
          >
            <item.icon className={cn("w-4 h-4", item.available ? item.color : "text-muted-foreground")} />
            <span className="text-xs flex-1 text-left">{item.label}</span>
            {item.available && <Check className="w-3 h-3 text-accent" />}
          </Button>
        ))}
      </div>
    </div>
  );
};

export default DownloadPanel;
