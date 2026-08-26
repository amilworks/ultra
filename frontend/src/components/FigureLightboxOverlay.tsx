import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Dialog as DialogPrimitive } from "radix-ui";
import { ChevronLeft, ChevronRight, Columns2, Download, Layers, Minus, Plus, X } from "lucide-react";

import { type LightboxFigure } from "@/lib/figureLightbox";
import { getLensOpener } from "@/lib/lensNavigation";

const MIN_SCALE = 1;
const MAX_SCALE = 8;

type Transform = { scale: number; x: number; y: number };
const IDENTITY: Transform = { scale: 1, x: 0, y: 0 };

const clamp = (value: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, value));

type Props = {
  figures: LightboxFigure[];
  initialIndex: number;
  onClose: () => void;
};

// A calm, in-app scientific figure viewer: zoom + pan at native resolution,
// prev/next + filmstrip across the run's figures, and a side-by-side compare mode.
// Replaces the old "open in a new tab" jump.
export function FigureLightboxOverlay({ figures, initialIndex, onClose }: Props) {
  const [index, setIndex] = useState(initialIndex);
  const [pinIndex, setPinIndex] = useState(() => (initialIndex + 1) % Math.max(1, figures.length));
  const [compare, setCompare] = useState(false);
  const [transform, setTransform] = useState<Transform>(IDENTITY);

  const stageRef = useRef<HTMLDivElement>(null);
  const transformRef = useRef(transform);
  useEffect(() => {
    transformRef.current = transform;
  }, [transform]);

  const total = figures.length;
  const active = figures[index];
  const pinned = figures[pinIndex] ?? figures[(index + 1) % total];
  // Read at render time so the button only exists while App has an opener
  // registered — never a dead control (the same registry the chat pills use).
  const openInLens = getLensOpener();

  // Reset the zoom/pan whenever the focused figure or the mode changes (done in
  // the handlers below rather than an effect, to avoid a cascading-render reset).
  const goTo = useCallback(
    (next: number) => {
      setTransform(IDENTITY);
      setIndex((current) => {
        const wrapped = ((next % total) + total) % total;
        if (compare && wrapped !== current) {
          setPinIndex(current); // push the figure we were on into the compare slot
        }
        return wrapped;
      });
    },
    [total, compare]
  );

  const toggleCompare = useCallback(() => {
    setTransform(IDENTITY);
    setCompare((value) => !value);
  }, []);

  const zoomAround = useCallback((factor: number, clientX: number, clientY: number) => {
    const node = stageRef.current;
    if (!node) {
      return;
    }
    const rect = node.getBoundingClientRect();
    const { scale, x, y } = transformRef.current;
    const next = clamp(scale * factor, MIN_SCALE, MAX_SCALE);
    if (next === MIN_SCALE) {
      setTransform(IDENTITY);
      return;
    }
    const ratio = next / scale;
    const px = clientX - rect.left - rect.width / 2;
    const py = clientY - rect.top - rect.height / 2;
    setTransform({ scale: next, x: px - (px - x) * ratio, y: py - (py - y) * ratio });
  }, []);

  const zoomCenter = useCallback((factor: number) => {
    const node = stageRef.current;
    if (!node) {
      return;
    }
    const rect = node.getBoundingClientRect();
    zoomAround(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
  }, [zoomAround]);

  // Wheel + iOS pinch (gesture*) zoom — attached non-passive so preventDefault works.
  useEffect(() => {
    const node = stageRef.current;
    if (!node) {
      return;
    }
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoomAround(event.deltaY < 0 ? 1.15 : 1 / 1.15, event.clientX, event.clientY);
    };
    let gestureBase = 1;
    const onGestureStart = (event: Event) => {
      event.preventDefault();
      gestureBase = transformRef.current.scale;
    };
    const onGestureChange = (event: Event) => {
      event.preventDefault();
      const scale = clamp(gestureBase * (event as unknown as { scale: number }).scale, MIN_SCALE, MAX_SCALE);
      zoomCenter(scale / transformRef.current.scale);
    };
    const prevent = (event: Event) => event.preventDefault();
    node.addEventListener("wheel", onWheel, { passive: false });
    node.addEventListener("gesturestart", onGestureStart, { passive: false });
    node.addEventListener("gesturechange", onGestureChange, { passive: false });
    node.addEventListener("gestureend", prevent, { passive: false });
    return () => {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("gesturestart", onGestureStart);
      node.removeEventListener("gesturechange", onGestureChange);
      node.removeEventListener("gestureend", prevent);
    };
  }, [zoomAround, zoomCenter]);

  // Keyboard: arrows navigate, +/- zoom, c toggles compare.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") {
        goTo(index - 1);
      } else if (event.key === "ArrowRight") {
        goTo(index + 1);
      } else if (event.key === "+" || event.key === "=") {
        zoomCenter(1.25);
      } else if (event.key === "-") {
        zoomCenter(1 / 1.25);
      } else if (event.key.toLowerCase() === "c") {
        toggleCompare();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, goTo, zoomCenter, toggleCompare]);

  // Pointer drag to pan when zoomed in.
  const dragRef = useRef<{ startX: number; startY: number; origin: Transform } | null>(null);
  const onPointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (transformRef.current.scale <= MIN_SCALE) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { startX: event.clientX, startY: event.clientY, origin: transformRef.current };
  }, []);
  const onPointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    setTransform({
      scale: drag.origin.scale,
      x: drag.origin.x + (event.clientX - drag.startX),
      y: drag.origin.y + (event.clientY - drag.startY),
    });
  }, []);
  const endPointer = useCallback(() => {
    dragRef.current = null;
  }, []);

  const onDoubleClick = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (transformRef.current.scale > MIN_SCALE) {
        setTransform(IDENTITY);
      } else {
        zoomAround(2.4, event.clientX, event.clientY);
      }
    },
    [zoomAround]
  );

  const transformStyle = useMemo(
    () => ({ transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})` }),
    [transform]
  );

  const zoomPct = Math.round(transform.scale * 100);
  const title = compare ? `${active?.title ?? ""}  vs  ${pinned?.title ?? ""}` : active?.title ?? "Figure";

  return (
    <DialogPrimitive.Root open onOpenChange={(open) => !open && onClose()}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="figure-lightbox-backdrop" />
        <DialogPrimitive.Content className="figure-lightbox" aria-describedby={undefined}>
          <DialogPrimitive.Title className="sr-only">{title}</DialogPrimitive.Title>

          <div className="figure-lightbox-bar">
            <span className="figure-lightbox-title" title={title}>
              {title}
            </span>
            <span className="figure-lightbox-count">
              {index + 1} / {total}
            </span>
            <span className="figure-lightbox-bar-spacer" />
            <button type="button" className="figure-lightbox-act" aria-label="Zoom out" onClick={() => zoomCenter(1 / 1.25)}>
              <Minus aria-hidden="true" />
            </button>
            <span className="figure-lightbox-zoom">{zoomPct}%</span>
            <button type="button" className="figure-lightbox-act" aria-label="Zoom in" onClick={() => zoomCenter(1.25)}>
              <Plus aria-hidden="true" />
            </button>
            <span className="figure-lightbox-sep" />
            {total > 1 ? (
              <button
                type="button"
                className="figure-lightbox-act figure-lightbox-act-labeled"
                aria-pressed={compare}
                onClick={() => toggleCompare()}
              >
                <Columns2 aria-hidden="true" />
                <span>Compare</span>
              </button>
            ) : null}
            {active?.fileId && openInLens ? (
              <button
                type="button"
                className="figure-lightbox-act"
                aria-label="Open in Lens"
                title="Open in Lens"
                onClick={() => {
                  openInLens([active.fileId as string]);
                  onClose();
                }}
              >
                <Layers aria-hidden="true" />
              </button>
            ) : null}
            <a
              className="figure-lightbox-act"
              href={active?.downloadUrl ?? active?.url}
              download
              aria-label="Download figure"
              title="Download"
            >
              <Download aria-hidden="true" />
            </a>
            <button type="button" className="figure-lightbox-act" aria-label="Close" onClick={onClose}>
              <X aria-hidden="true" />
            </button>
          </div>

          <div
            className="figure-lightbox-stage"
            ref={stageRef}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endPointer}
            onPointerCancel={endPointer}
            onDoubleClick={onDoubleClick}
            data-grabbable={transform.scale > MIN_SCALE ? "true" : undefined}
          >
            {total > 1 ? (
              <button
                type="button"
                className="figure-lightbox-nav figure-lightbox-prev"
                aria-label="Previous figure"
                onClick={() => goTo(index - 1)}
              >
                <ChevronLeft aria-hidden="true" />
              </button>
            ) : null}

            <div className="figure-lightbox-frames">
              {compare ? (
                <>
                  <FigurePane figure={pinned} transformStyle={transformStyle} />
                  <FigurePane figure={active} transformStyle={transformStyle} focused />
                </>
              ) : (
                <FigurePane figure={active} transformStyle={transformStyle} />
              )}
            </div>

            {total > 1 ? (
              <button
                type="button"
                className="figure-lightbox-nav figure-lightbox-next"
                aria-label="Next figure"
                onClick={() => goTo(index + 1)}
              >
                <ChevronRight aria-hidden="true" />
              </button>
            ) : null}
          </div>

          {total > 1 ? (
            <div className="figure-lightbox-strip">
              {figures.map((figure, figureIndex) => (
                <button
                  key={`${figure.url}-${figureIndex}`}
                  type="button"
                  className="figure-lightbox-thumb"
                  data-active={figureIndex === index ? "true" : undefined}
                  data-pinned={compare && figureIndex === pinIndex ? "true" : undefined}
                  title={figure.title}
                  aria-label={figure.title}
                  onClick={() => goTo(figureIndex)}
                >
                  {figure.url ? <img src={figure.url} alt="" loading="lazy" /> : null}
                </button>
              ))}
            </div>
          ) : null}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

function FigurePane({
  figure,
  transformStyle,
  focused,
}: {
  figure: LightboxFigure | undefined;
  transformStyle: { transform: string };
  focused?: boolean;
}) {
  if (!figure) {
    return null;
  }
  return (
    <div className="figure-lightbox-pane" data-focused={focused ? "true" : undefined}>
      <img
        className="figure-lightbox-img"
        src={figure.url}
        alt={figure.title}
        style={transformStyle}
        draggable={false}
      />
      <span className="figure-lightbox-pane-caption">{figure.title}</span>
    </div>
  );
}
