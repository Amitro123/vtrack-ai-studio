import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Zap, Target } from "lucide-react";

interface ProcessingModeToggleProps {
    mode: "fast" | "accurate";
    onModeChange: (mode: "fast" | "accurate") => void;
    disabled?: boolean;
}

const ProcessingModeToggle = ({ mode, onModeChange, disabled = false }: ProcessingModeToggleProps) => {
    return (
        <div className="glass-panel p-4 rounded-lg border border-border/50">
            <Label className="text-sm font-medium mb-3 block">Processing Mode</Label>
            <RadioGroup
                value={mode}
                onValueChange={(value) => onModeChange(value as "fast" | "accurate")}
                disabled={disabled}
                className="flex gap-3"
            >
                <div className="flex items-center space-x-2 flex-1">
                    <RadioGroupItem value="fast" id="mode-fast" />
                    <Label
                        htmlFor="mode-fast"
                        className="flex items-center gap-2 cursor-pointer text-sm"
                    >
                        <Zap className="w-4 h-4 text-yellow-500" />
                        <div>
                            <div className="font-medium">Fast</div>
                            <div className="text-xs text-muted-foreground">
                                5 FPS, 12 keyframes
                            </div>
                        </div>
                    </Label>
                </div>
                <div className="flex items-center space-x-2 flex-1">
                    <RadioGroupItem value="accurate" id="mode-accurate" />
                    <Label
                        htmlFor="mode-accurate"
                        className="flex items-center gap-2 cursor-pointer text-sm"
                    >
                        <Target className="w-4 h-4 text-blue-500" />
                        <div>
                            <div className="font-medium">Accurate</div>
                            <div className="text-xs text-muted-foreground">
                                10 FPS, 32 keyframes
                            </div>
                        </div>
                    </Label>
                </div>
            </RadioGroup>
            <p className="text-xs text-muted-foreground mt-3">
                {mode === "fast"
                    ? "⚡ Faster processing - Good for quick previews"
                    : "🎯 Higher quality - Better temporal accuracy"}
            </p>
        </div>
    );
};

export default ProcessingModeToggle;
