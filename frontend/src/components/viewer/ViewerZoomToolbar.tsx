import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";
import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

type ViewerZoomToolbarProps = {
  /** Zoom readout text ("Fit" / "73%" / "150%"), already formatted. */
  readout: string;
  onZoomOut: () => void;
  onZoomIn: () => void;
  onFit: () => void;
  /** Optional leading readout (e.g. the pixel-dimension badge) shown before the zoom %. */
  extraReadout?: ReactNode;
};

// The single source of truth for the 2D viewer's on-canvas controls, so the
// direct-plane and deep-zoom surfaces present identical affordances and labels:
// an optional context badge, the zoom readout, then Zoom out / Zoom in / Fit.
// Marked data-viewer-image-control so the canvases ignore pointerdowns on it.
export function ViewerZoomToolbar({ readout, onZoomOut, onZoomIn, onFit, extraReadout }: ViewerZoomToolbarProps) {
  return (
    <div className="viewer-direct-image-toolbar viewer-chrome-fade" data-viewer-image-control="true">
      {extraReadout ? (
        <Badge variant="outline" className="viewer-direct-image-readout">
          {extraReadout}
        </Badge>
      ) : null}
      <Badge variant="secondary" className="viewer-direct-image-readout" data-testid="viewer-zoom-readout">
        {readout}
      </Badge>
      <Button type="button" variant="outline" size="icon-xs" aria-label="Zoom out" onClick={onZoomOut}>
        <ZoomOut data-icon="inline-start" />
      </Button>
      <Button type="button" variant="outline" size="icon-xs" aria-label="Zoom in" onClick={onZoomIn}>
        <ZoomIn data-icon="inline-start" />
      </Button>
      <Button type="button" variant="outline" size="icon-xs" aria-label="Fit image to view" onClick={onFit}>
        <Maximize2 data-icon="inline-start" />
      </Button>
    </div>
  );
}
