import { Sparkles, Video, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const Header = () => {
  const [device, setDevice] = useState<string>("cpu");

  useEffect(() => {
    api.getHealth().then((health) => {
      setDevice(health.device || "cpu");
    }).catch(() => {
      // Ignore errors
    });
  }, []);

  return (
    <header className="glass-panel border-b border-border/50 px-6 py-4">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center">
              <Video className="w-5 h-5 text-primary-foreground" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent rounded-full animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              <span className="gradient-text">VTrackAI</span>
            </h1>
            <p className="text-xs text-muted-foreground font-mono">Click to Isolate Sound</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/50 border border-border/50">
            <Zap className="w-3.5 h-3.5 text-accent" />
            <span className="text-xs text-muted-foreground font-mono">
              {device === "cuda" ? "🔥 GPU Ready" : "💻 CPU Mode"}
            </span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 border border-primary/30">
            <Sparkles className="w-3.5 h-3.5 text-primary" />
            <span className="text-xs text-primary font-medium">SAM3</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
