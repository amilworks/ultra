import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

type SettingsPanelProps = {
  apiBaseUrl: string;
  apiKey: string;
  maxToolCalls: number;
  maxRuntimeSeconds: number;
  scratchpad: boolean;
  showDiagnostics: boolean;
  autoReproReport: boolean;
  includeFileContext: boolean;
  healthStatus?: string | null;
  onApiBaseUrlChange: (value: string) => void;
  onApiKeyChange: (value: string) => void;
  onMaxToolCallsChange: (value: number) => void;
  onMaxRuntimeSecondsChange: (value: number) => void;
  onScratchpadChange: (value: boolean) => void;
  onShowDiagnosticsChange: (value: boolean) => void;
  onAutoReproReportChange: (value: boolean) => void;
  onIncludeFileContextChange: (value: boolean) => void;
  onTestConnection: () => Promise<void>;
};

export function SettingsPanel(props: SettingsPanelProps) {
  return (
    <Card className="border-white/60 bg-white/82 shadow-xl backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-3">
        <CardTitle>Session config</CardTitle>
        <Button variant="outline" size="sm" onClick={() => void props.onTestConnection()}>
          Test API
        </Button>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {props.healthStatus ? (
          <p className="text-muted-foreground text-sm">{props.healthStatus}</p>
        ) : null}
        <label className="grid gap-1.5">
          <span className="text-muted-foreground text-xs">API base URL</span>
          <Input
            value={props.apiBaseUrl}
            onChange={(event) => props.onApiBaseUrlChange(event.target.value)}
            placeholder="http://127.0.0.1:8000"
          />
        </label>
        <label className="grid gap-1.5">
          <span className="text-muted-foreground text-xs">API key (optional)</span>
          <Input
            value={props.apiKey}
            onChange={(event) => props.onApiKeyChange(event.target.value)}
            placeholder="X-API-Key"
          />
        </label>
        <div className="grid grid-cols-2 gap-2">
          <label className="grid gap-1.5">
            <span className="text-muted-foreground text-xs">Max tool calls</span>
            <Input
              type="number"
              min={1}
              max={64}
              value={props.maxToolCalls}
              onChange={(event) => props.onMaxToolCallsChange(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-1.5">
            <span className="text-muted-foreground text-xs">Runtime (seconds)</span>
            <Input
              type="number"
              min={1}
              max={86400}
              value={props.maxRuntimeSeconds}
              onChange={(event) => props.onMaxRuntimeSecondsChange(Number(event.target.value))}
            />
          </label>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <Switch
            checked={props.scratchpad}
            onCheckedChange={(checked) => props.onScratchpadChange(Boolean(checked))}
          />
          <span>Enable scratchpad artifact</span>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Switch
            checked={props.includeFileContext}
            onCheckedChange={(checked) =>
              props.onIncludeFileContextChange(Boolean(checked))
            }
          />
          <span>Inject file context system prompt</span>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Switch
            checked={props.showDiagnostics}
            onCheckedChange={(checked) => props.onShowDiagnosticsChange(Boolean(checked))}
          />
          <span>Show diagnostics and tool traces</span>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Switch
            checked={props.autoReproReport}
            onCheckedChange={(checked) => props.onAutoReproReportChange(Boolean(checked))}
          />
          <span>Auto-generate reproducibility report</span>
        </label>
      </CardContent>
    </Card>
  );
}
