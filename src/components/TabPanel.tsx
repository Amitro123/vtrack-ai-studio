import { useState } from "react";
import { MousePointer2, MessageSquare, Eraser, Send, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface TabPanelProps {
  activeTab: "click" | "chat" | "remove";
  onTabChange: (tab: "click" | "chat" | "remove") => void;
  onChatSubmit: (message: string) => void;
  onRemoveAction: () => void;
  isProcessing: boolean;
  chatMessages: Array<{ role: "user" | "ai"; content: string }>;
  hasVideo: boolean;
}

const TabPanel = ({
  activeTab,
  onTabChange,
  onChatSubmit,
  onRemoveAction,
  isProcessing,
  chatMessages,
  hasVideo,
}: TabPanelProps) => {
  const [chatInput, setChatInput] = useState("");

  const tabs = [
    { id: "click" as const, label: "Click", icon: MousePointer2, description: "Click to track objects" },
    { id: "chat" as const, label: "Chat", icon: MessageSquare, description: "Natural language isolation" },
    { id: "remove" as const, label: "Remove", icon: Eraser, description: "Remove & inpaint" },
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (chatInput.trim() && !isProcessing) {
      onChatSubmit(chatInput.trim());
      setChatInput("");
    }
  };

  return (
    <div className="flex flex-col h-full glass-panel rounded-xl overflow-hidden">
      {/* Tab headers */}
      <div className="flex border-b border-border/50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={cn(
              "flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-all duration-200",
              activeTab === tab.id
                ? "text-primary border-b-2 border-primary bg-primary/5"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary/30"
            )}
          >
            <tab.icon className="w-4 h-4" />
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 flex flex-col p-4 overflow-hidden">
        {activeTab === "click" && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-4 animate-fade-in">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center">
              <MousePointer2 className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Click to Track</h3>
              <p className="text-sm text-muted-foreground mt-1">
                {hasVideo
                  ? "Click on any object in the video to generate a SAM2 mask and track it across frames"
                  : "Upload a video first, then click on objects to track"}
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className="w-3 h-3 rounded-full bg-primary" />
              <span>Red mask = tracked object</span>
            </div>
          </div>
        )}

        {activeTab === "chat" && (
          <div className="flex flex-col h-full animate-fade-in">
            {/* Chat messages */}
            <div className="flex-1 overflow-y-auto space-y-3 mb-4">
              {chatMessages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent/20 to-accent/10 flex items-center justify-center">
                    <MessageSquare className="w-8 h-8 text-accent" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">Semantic Isolation</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Describe what to isolate: "Isolate drums", "Extract vocals", "Find guitar"
                    </p>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2">
                    {["Isolate drums", "Extract vocals", "Find the piano"].map((example) => (
                      <button
                        key={example}
                        onClick={() => setChatInput(example)}
                        className="px-3 py-1.5 text-xs rounded-full bg-secondary/50 hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                chatMessages.map((msg, i) => (
                  <div
                    key={i}
                    className={cn(
                      "flex gap-2",
                      msg.role === "user" ? "justify-end" : "justify-start"
                    )}
                  >
                    <div
                      className={cn(
                        "max-w-[80%] px-4 py-2 rounded-2xl text-sm",
                        msg.role === "user"
                          ? "bg-primary text-primary-foreground rounded-tr-none"
                          : "bg-secondary text-secondary-foreground rounded-tl-none"
                      )}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))
              )}
              {isProcessing && (
                <div className="flex gap-2">
                  <div className="px-4 py-2 rounded-2xl rounded-tl-none bg-secondary">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-accent" />
                      <span className="text-sm text-muted-foreground">
                        Processing with GroundingDINO + Demucs...
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Chat input */}
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder={hasVideo ? "Describe what to isolate..." : "Upload a video first..."}
                disabled={!hasVideo || isProcessing}
                className="flex-1 px-4 py-2.5 rounded-lg bg-secondary/50 border border-border/50 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <Button
                type="submit"
                disabled={!hasVideo || !chatInput.trim() || isProcessing}
                className="shrink-0"
              >
                <Send className="w-4 h-4" />
              </Button>
            </form>
          </div>
        )}

        {activeTab === "remove" && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-4 animate-fade-in">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-destructive/20 to-destructive/10 flex items-center justify-center">
              <Eraser className="w-8 h-8 text-destructive" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Object Removal</h3>
              <p className="text-sm text-muted-foreground mt-1">
                {hasVideo
                  ? "Select an object to track, then remove it using ProPainter inpainting"
                  : "Upload a video first to remove objects"}
              </p>
            </div>
            <Button
              variant="destructive"
              onClick={onRemoveAction}
              disabled={!hasVideo || isProcessing}
              className="gap-2"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Eraser className="w-4 h-4" />
                  Remove Selected Object
                </>
              )}
            </Button>
            <p className="text-xs text-muted-foreground">
              Uses SAM2 tracking + ProPainter
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TabPanel;
