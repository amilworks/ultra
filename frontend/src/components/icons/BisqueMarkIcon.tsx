import type { SVGProps } from "react";

export function BisqueMarkIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 128 128"
      fill="none"
      aria-hidden="true"
      {...props}
    >
      <g stroke="currentColor" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 32V64A44 44 0 0 0 108 64V32" />
        <path d="M34 58A16 16 0 0 1 64 48A16 16 0 0 0 94 58" />
      </g>
    </svg>
  );
}
