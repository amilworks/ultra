import { Suspense, lazy, useEffect, useState, type ComponentType } from "react";
import { loadSonnerModule } from "@/lib/toast";

const LazyToaster = lazy(async () => {
  const module = await loadSonnerModule();
  return {
    default: module.Toaster as ComponentType<any>,
  };
});

export function DeferredToaster({ theme }: { theme: "light" | "dark" }) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const show = () => setReady(true);
    if (typeof window.requestIdleCallback === "function") {
      const idleId = window.requestIdleCallback(show, { timeout: 3_000 });
      return () => window.cancelIdleCallback(idleId);
    }

    const timeoutId = window.setTimeout(show, 1_200);
    return () => window.clearTimeout(timeoutId);
  }, []);

  if (!ready) {
    return null;
  }

  return (
    <Suspense fallback={null}>
      <LazyToaster
        theme={theme}
        richColors
        position="bottom-right"
        toastOptions={{ duration: 4200 }}
      />
    </Suspense>
  );
}
