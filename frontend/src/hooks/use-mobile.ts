import * as React from "react";

const MOBILE_BREAKPOINT = 768;
const getServerSnapshot = (): boolean => false;

export function useIsMobile() {
  const subscribe = React.useCallback((onStoreChange: () => void) => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`);
    mql.addEventListener("change", onStoreChange);
    return () => mql.removeEventListener("change", onStoreChange);
  }, []);

  const getSnapshot = React.useCallback(
    () => window.innerWidth < MOBILE_BREAKPOINT,
    []
  );

  return React.useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
