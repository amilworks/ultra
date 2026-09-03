import type { SVGProps } from "react";

const navigationIconClassName = (name: string, className?: string): string =>
  ["meridian-nav-icon", name, className].filter(Boolean).join(" ");

/* The Meridian symbol set — small instrument drawings, not illustrations.
   Stroke-only, currentColor, one weight, so every mark survives being stamped
   at 16px in a single colour. One symbol is wired:

     Trace    — a record being written: the streaming/thinking indicator. It
                accumulates left to right and holds; it never loops, because a
                reasoning trace is a record, not a spinner passing time.

   Two are RESERVED — defined so the next surface that needs them uses these
   instead of inventing new marks, deliberately unwired until then:

     Reticle  — a fiducial cross in an aperture: alignment, calibration,
                "measured against". (It briefly served as the app mark; the
                brand stays on the BisQue glyph by decision — the product's
                lineage outranks the language's own iconography.)
     Transit  — a crossing of the reference line: checkpoints, run
                milestones, timestamps.

   The fourth symbol, the point of light, is not an icon — it is drawn with
   --accent-live (see .running-status-point and MeridianField). */

export function ReticleIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      aria-hidden="true"
      {...props}
    >
      {/* Ticks deliberately cross the rim — a fiducial is drawn through its
          aperture, not butted against it. */}
      <circle cx="12" cy="12" r="7.4" />
      <path d="M12 1.6v6.4M12 16v6.4M1.6 12H8M16 12h6.4" />
      <circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function TraceIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 40 12"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.3}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {/* pathLength=1 normalises stroke-dash animation regardless of the
          real geometry, so the CSS write-on needs no magic numbers. */}
      <path pathLength={1} d="M1 8h5l2.5-4 2.5 7 2.5-9 2.5 5.5 2.5-3 2.5 2h3" />
    </svg>
  );
}

/* The recorder variant of the trace: built to replace the opening segment of
   a hairline, not to sit beside it. The path begins at (0, 5), returns exactly
   to y=5 after the measured excursion, and ends at (96, 5). Butt end caps keep
   those endpoints coincident with the 96x10 coordinate box; rendered at
   6rem x 10px, its units map 1:1. */
export function RecorderTraceIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 96 10"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.3}
      strokeLinecap="butt"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path
        pathLength={1}
        d="M0 5H18L21 1.6L24 7.6L27 0.6L30 5.6L33 3L36 5H96"
      />
    </svg>
  );
}

export function TransitIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      aria-hidden="true"
      {...props}
    >
      <path d="M12 3v18" />
      <path d="M9 6h6M9 18h6" opacity={0.45} />
      <path d="M6 12h12" strokeWidth={1.8} />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
    </svg>
  );
}

/* Navigation marks keep Lucide's immediate recognition but separate the parts
   that carry meaning in motion. The folder swaps one measured silhouette for
   the other; Notes moves only the pen and the ink it leaves behind. */
export function MeridianFolderIcon({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      data-meridian-icon="folder"
      className={navigationIconClassName("meridian-folder-icon", className)}
      {...props}
    >
      <path
        className="meridian-folder-closed"
        d="M20.25 19.25A1.75 1.75 0 0 0 22 17.5V8.25a1.75 1.75 0 0 0-1.75-1.75h-7.7a2 2 0 0 1-1.68-.92l-.78-1.2a2 2 0 0 0-1.68-.92H4.75A1.75 1.75 0 0 0 3 5.21V17.5c0 .97.78 1.75 1.75 1.75Z"
      />
      <path
        className="meridian-folder-open"
        d="M5.45 13.55l1.4-2.7a1.8 1.8 0 0 1 1.6-.98h11.42a1.8 1.8 0 0 1 1.74 2.25l-1.46 5.53a2.1 2.1 0 0 1-2.03 1.57H4.8A1.8 1.8 0 0 1 3 17.42V5.21c0-.97.78-1.75 1.75-1.75h3.66a2 2 0 0 1 1.68.92l.78 1.2a2 2 0 0 0 1.68.92h5.95c.97 0 1.75.78 1.75 1.75v1.02"
      />
    </svg>
  );
}

export function MeridianNotesIcon({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      data-meridian-icon="notes"
      className={navigationIconClassName("meridian-notes-icon", className)}
      {...props}
    >
      <path d="M13.4 2.25H6.1A1.85 1.85 0 0 0 4.25 4.1v15.8c0 1.02.83 1.85 1.85 1.85h11.8c1.02 0 1.85-.83 1.85-1.85v-7.25" />
      <path d="M2.25 6.25h4M2.25 10.1h4M2.25 13.95h4M2.25 17.8h4" />
      <path
        className="meridian-notes-ink"
        pathLength={1}
        d="M12.25 12.35c-1.35.2-2.55.8-3.35 1.72"
      />
      <path
        className="meridian-notes-pen"
        d="M21.25 5.75a1.05 1.05 0 0 0-2.97-2.97l-4.85 4.86a2 2 0 0 0-.5.84l-.8 2.73a.5.5 0 0 0 .62.62l2.73-.8a2 2 0 0 0 .84-.5Z"
      />
    </svg>
  );
}

/* File identity marks use the same measured stroke and round joins as the
   instrument symbols above. The folded sheet says “stored artifact”; the
   internal mark says what kind of evidence it contains without borrowing a
   vendor or language logo. */
export function MeridianFileIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      data-meridian-icon="file"
      {...props}
    >
      <path d="M6.75 3.25h6.5l4.5 4.5v13H6.75z" />
      <path d="M13.25 3.25v4.5h4.5" />
      <circle cx="12.25" cy="14.25" r="2.2" />
      <path d="M12.25 10.65v1.4M12.25 16.45v1.4M8.65 14.25h1.4M14.45 14.25h1.4" />
      <circle cx="12.25" cy="14.25" r="0.6" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function MeridianSourceFileIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      data-meridian-icon="source-file"
      {...props}
    >
      <path d="M6.75 3.25h6.5l4.5 4.5v13H6.75z" />
      <path d="M13.25 3.25v4.5h4.5" />
      <path d="M8.75 15.25h1.55l1.05-2.55 1.35 4.05 1.15-2.55h1.65" />
    </svg>
  );
}
