type FontLoader = Pick<FontFaceSet, "load">;

type CanvasFontRequest = {
  query: string;
  sample: string;
};

export function scheduleFontsReadyRedraw(
  fonts: FontLoader,
  requests: CanvasFontRequest[],
  redraw: () => void
): () => void {
  let mounted = true;
  void Promise.all(
    requests.map(({ query, sample }) => fonts.load(query, sample))
  ).then(
    () => {
      if (mounted) {
        redraw();
      }
    },
    () => {
      // A failed font load keeps the already-painted fallback canvas. The
      // browser smoke contract reports the failed request separately.
    }
  );
  return () => {
    mounted = false;
  };
}
