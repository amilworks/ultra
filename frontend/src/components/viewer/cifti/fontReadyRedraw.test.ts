import { describe, expect, it, vi } from "vitest";

import { scheduleFontsReadyRedraw } from "./fontReadyRedraw";

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
};

describe("scheduleFontsReadyRedraw", () => {
  it("waits for every requested family and then redraws exactly once", async () => {
    const ultraSansLoad = deferred<FontFace[]>();
    const monoLoad = deferred<FontFace[]>();
    const load = vi.fn((query: string) =>
      query.includes("Ultra Mono") ? monoLoad.promise : ultraSansLoad.promise
    );
    const redraw = vi.fn();

    scheduleFontsReadyRedraw(
      { load } as unknown as Pick<FontFaceSet, "load">,
      [
        {
          query: '600 11px "BisQue Ultra Sans"',
          sample: "CORTEX_LEFT",
        },
        {
          query: '400 11px "BisQue Ultra Mono"',
          sample: "frame index 12,345",
        },
      ],
      redraw
    );
    expect(load).toHaveBeenCalledTimes(2);
    expect(load).toHaveBeenNthCalledWith(
      1,
      '600 11px "BisQue Ultra Sans"',
      "CORTEX_LEFT"
    );
    expect(load).toHaveBeenNthCalledWith(
      2,
      '400 11px "BisQue Ultra Mono"',
      "frame index 12,345"
    );

    ultraSansLoad.resolve([]);
    await ultraSansLoad.promise;
    await Promise.resolve();
    expect(redraw).not.toHaveBeenCalled();

    monoLoad.resolve([]);
    await monoLoad.promise;
    await Promise.resolve();
    expect(redraw).toHaveBeenCalledTimes(1);
    await Promise.resolve();
    expect(redraw).toHaveBeenCalledTimes(1);
  });

  it("does not redraw after unmount while any requested face is pending", async () => {
    const firstLoad = deferred<FontFace[]>();
    const secondLoad = deferred<FontFace[]>();
    const load = vi
      .fn()
      .mockReturnValueOnce(firstLoad.promise)
      .mockReturnValueOnce(secondLoad.promise);
    const redraw = vi.fn();
    const cleanup = scheduleFontsReadyRedraw(
      { load } as unknown as Pick<FontFaceSet, "load">,
      [
        {
          query: '600 10px "BisQue Ultra Sans"',
          sample: "CORTEX_LEFT",
        },
        {
          query: '400 10px "BisQue Ultra Mono"',
          sample: "+0.75 0 −0.75",
        },
      ],
      redraw
    );

    firstLoad.resolve([]);
    await firstLoad.promise;
    cleanup();
    secondLoad.resolve([]);
    await secondLoad.promise;
    await Promise.resolve();
    expect(redraw).not.toHaveBeenCalled();
  });
});
