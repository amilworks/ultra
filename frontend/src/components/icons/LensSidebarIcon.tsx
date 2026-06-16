import type { SVGProps } from "react";
import { Layers3 } from "lucide-react";

type LensSidebarIconProps = SVGProps<SVGSVGElement> & {
  active?: boolean;
};

export function LensSidebarIcon({ active = false, className, ...props }: LensSidebarIconProps) {
  return (
    <Layers3
      {...props}
      className={["app-lens-sidebar-icon", className].filter(Boolean).join(" ")}
      data-lens-icon={active ? "active" : "default"}
    />
  );
}
