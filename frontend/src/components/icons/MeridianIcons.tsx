import type { SVGProps } from "react";

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
