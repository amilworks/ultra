import { describe, expect, it } from "vitest";

import { buildBisqueNavLinks, inferBisqueRootFromUrl } from "./bisqueNavigation";

describe("bisqueNavigation", () => {
  it("normalizes the BisQue root into navigation links", () => {
    expect(buildBisqueNavLinks("https://bisque.example.org/")).toEqual({
      home: "https://bisque.example.org/client_service/",
      datasets:
        "https://bisque.example.org/client_service/browser?resource=/data_service/dataset",
      images:
        "https://bisque.example.org/client_service/browser?resource=/data_service/image",
      tables:
        "https://bisque.example.org/client_service/browser?resource=/data_service/table",
    });
  });

  it("infers the origin from a full BisQue URL", () => {
    expect(
      inferBisqueRootFromUrl(
        "https://bisque.example.org/client_service/browser?resource=/data_service/image"
      )
    ).toBe("https://bisque.example.org");
  });
});
