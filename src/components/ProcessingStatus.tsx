import { Loader2, Check, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ProcessingStep {
  id: string;
  label: string;
  status: "pending" | "processing" | "complete" | "error";
}

interface ProcessingStatusProps {
  steps: ProcessingStep[];
  visible: boolean;
}

const ProcessingStatus = ({ steps, visible }: ProcessingStatusProps) => {
  if (!visible) return null;

  return (
    <div className="glass-panel rounded-xl p-4 animate-fade-in">
      <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin text-primary" />
        Processing Pipeline
      </h3>

      <div className="space-y-2">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={cn(
              "flex items-center gap-3 p-2 rounded-lg transition-colors",
              step.status === "processing" && "bg-primary/10",
              step.status === "complete" && "bg-accent/10",
              step.status === "error" && "bg-destructive/10"
            )}
          >
            <div className="w-6 h-6 rounded-full flex items-center justify-center shrink-0">
              {step.status === "pending" && (
                <span className="text-xs text-muted-foreground font-mono">{index + 1}</span>
              )}
              {step.status === "processing" && (
                <Loader2 className="w-4 h-4 animate-spin text-primary" />
              )}
              {step.status === "complete" && (
                <Check className="w-4 h-4 text-accent" />
              )}
              {step.status === "error" && (
                <AlertCircle className="w-4 h-4 text-destructive" />
              )}
            </div>
            <span
              className={cn(
                "text-sm",
                step.status === "pending" && "text-muted-foreground",
                step.status === "processing" && "text-primary font-medium",
                step.status === "complete" && "text-accent",
                step.status === "error" && "text-destructive"
              )}
            >
              {step.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProcessingStatus;
