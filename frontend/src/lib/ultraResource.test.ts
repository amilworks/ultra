import { describe, expect, it } from "vitest";

import { parseUltraResourceRef, ultraResourceRef } from "./ultraResource";

describe("parseUltraResourceRef", () => {
  it("round-trips an app-minted ref", () => {
    const ref = ultraResourceRef("file_abc", "my cells (2).tif");
    expect(parseUltraResourceRef(ref)).toEqual({ fileId: "file_abc", name: "my cells (2).tif" });
  });

  it("tolerates a name with a raw percent sign instead of throwing", () => {
    // Model-authored markdown: micromark keeps "%wt" as-is, which is not a valid
    // escape. A throw here would surface as a render error for the whole message.
    expect(() => parseUltraResourceRef("ultra://resource/file-3/5%wt_Ni.tif")).not.toThrow();
    expect(parseUltraResourceRef("ultra://resource/file-3/5%wt_Ni.tif")).toEqual({
      fileId: "file-3",
      name: "5%wt_Ni.tif",
    });
  });

  it("still decodes a well-formed escape", () => {
    expect(parseUltraResourceRef("ultra://resource/file-3/r%C3%A9sum%C3%A9.pdf")).toEqual({
      fileId: "file-3",
      name: "r\u00e9sum\u00e9.pdf",
    });
  });

  it("accepts a ref without a name and rejects non-refs", () => {
    expect(parseUltraResourceRef("ultra://resource/file-3")).toEqual({ fileId: "file-3", name: "" });
    expect(parseUltraResourceRef("https://example.org/resource/file-3")).toBeNull();
    expect(parseUltraResourceRef("ultra://resource/")).toBeNull();
  });
});
