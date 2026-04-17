import * as React from "react";

const getServerSnapshot = (): boolean => false;

export function useBreakpoint(breakpoint: number) {
  const subscribe = React.useCallback(
    (onStoreChange: () => void) => {
      const mql = window.matchMedia(`(max-width: ${breakpoint - 1}px)`);
      mql.addEventListener("change", onStoreChange);
      return () => mql.removeEventListener("change", onStoreChange);
    },
    [breakpoint]
  );

  const getSnapshot = React.useCallback(
    () => window.innerWidth < breakpoint,
    [breakpoint]
  );

  return React.useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
