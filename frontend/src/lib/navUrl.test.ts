import { describe, expect, it } from "vitest";

import {
  buildNavUrl,
  LENS_MAX_FILE_IDS,
  lensDeepLinkFor,
  navStateKey,
  normalizeLensFileIds,
  parseNavFromSearch,
  resolveLensLink,
  type NavState,
} from "./navUrl";

const loc = (search: string, pathname = "/", hash = "") => ({ pathname, search, hash });

const nav = (partial: Partial<NavState> & Pick<NavState, "panel">): NavState => ({
  resourceFileIds: [],
  resourceCollectionId: null,
  ...partial,
});

const manyIds = (count: number): string[] => Array.from({ length: count }, (_, index) => `id-${index}`);

describe("normalizeLensFileIds", () => {
  it("trims and drops empty ids", () => {
    expect(normalizeLensFileIds([" a ", "", "   ", "b"])).toEqual(["a", "b"]);
  });

  it("dedupes, keeping the first occurrence's position", () => {
    expect(normalizeLensFileIds(["b", "a", "b", " a", "c"])).toEqual(["b", "a", "c"]);
  });

  it("caps at LENS_MAX_FILE_IDS, counting only surviving ids", () => {
    expect(LENS_MAX_FILE_IDS).toBe(24);
    expect(normalizeLensFileIds(manyIds(30))).toEqual(manyIds(24));
    // Duplicates and blanks do not consume slots.
    const padded = ["", "id-0", "id-0", ...manyIds(30)];
    expect(normalizeLensFileIds(padded)).toEqual(manyIds(24));
  });

  it("is idempotent and never returns its input", () => {
    const input = manyIds(5);
    const once = normalizeLensFileIds(input);
    expect(once).toEqual(input);
    expect(once).not.toBe(input);
    expect(normalizeLensFileIds(once)).toEqual(once);
  });
});

describe("navUrl", () => {
  describe("buildNavUrl", () => {
    it("encodes chat as the absence of a view param", () => {
      expect(buildNavUrl(loc(""), nav({ panel: "chat" }))).toBe("/");
      // Clears a stale view/resource when returning to chat.
      expect(buildNavUrl(loc("?view=lens&resource=abc"), nav({ panel: "chat" }))).toBe("/");
    });

    it("encodes the resources / admin / training panels", () => {
      expect(buildNavUrl(loc(""), nav({ panel: "resources" }))).toBe("/?view=resources");
      expect(buildNavUrl(loc(""), nav({ panel: "admin" }))).toBe("/?view=admin");
      expect(buildNavUrl(loc(""), nav({ panel: "training" }))).toBe("/?view=training");
    });

    it("encodes Lens with its resource id(s)", () => {
      expect(buildNavUrl(loc(""), nav({ panel: "scientific-viewer", resourceFileIds: ["file-123"] }))).toBe(
        "/?view=lens&resource=file-123"
      );
      expect(
        buildNavUrl(loc(""), nav({ panel: "scientific-viewer", resourceFileIds: ["a", "b"] }))
      ).toBe("/?view=lens&resource=a%2Cb");
    });

    it("omits the resource param when there are no file ids", () => {
      expect(buildNavUrl(loc(""), nav({ panel: "scientific-viewer" }))).toBe("/?view=lens");
      expect(buildNavUrl(loc(""), nav({ panel: "scientific-viewer", resourceFileIds: ["", "  "] }))).toBe(
        "/?view=lens"
      );
    });

    it("writes the normalized id set (deduped, capped) so the URL matches what a restore parses", () => {
      expect(
        buildNavUrl(loc(""), nav({ panel: "scientific-viewer", resourceFileIds: ["a", " a", "b"] }))
      ).toBe("/?view=lens&resource=a%2Cb");
      const url = buildNavUrl(loc(""), nav({ panel: "scientific-viewer", resourceFileIds: manyIds(30) }));
      expect(parseNavFromSearch(url.slice(url.indexOf("?"))).resourceFileIds).toEqual(manyIds(24));
    });

    it("encodes the open Resources collection and clears it when leaving", () => {
      expect(
        buildNavUrl(loc(""), nav({ panel: "resources", resourceCollectionId: "col-42" }))
      ).toBe("/?view=resources&collection=col-42");
      // Back at the collection root: the param goes away.
      expect(
        buildNavUrl(loc("?view=resources&collection=col-42"), nav({ panel: "resources" }))
      ).toBe("/?view=resources");
      // A collection id never leaks onto a non-resources panel.
      expect(
        buildNavUrl(loc("?view=resources&collection=col-42"), nav({ panel: "chat", resourceCollectionId: "col-42" }))
      ).toBe("/");
    });

    it("PRESERVES other query params (e.g. conversation) so the two URL layers never clobber", () => {
      expect(
        buildNavUrl(loc("?conversation=conv-9"), nav({ panel: "scientific-viewer", resourceFileIds: ["file-1"] }))
      ).toBe("/?conversation=conv-9&view=lens&resource=file-1");
      // Switching to chat keeps the conversation but drops view/resource.
      expect(
        buildNavUrl(loc("?conversation=conv-9&view=lens&resource=file-1"), nav({ panel: "chat" }))
      ).toBe("/?conversation=conv-9");
    });

    it("preserves pathname and hash", () => {
      expect(buildNavUrl(loc("", "/code", "#section"), nav({ panel: "resources" }))).toBe(
        "/code?view=resources#section"
      );
    });
  });

  describe("parseNavFromSearch", () => {
    it("defaults to chat when no view param", () => {
      expect(parseNavFromSearch("")).toEqual(nav({ panel: "chat" }));
      expect(parseNavFromSearch("?conversation=conv-1")).toEqual(nav({ panel: "chat" }));
    });

    it("maps view -> panel", () => {
      expect(parseNavFromSearch("?view=resources").panel).toBe("resources");
      expect(parseNavFromSearch("?view=admin").panel).toBe("admin");
      expect(parseNavFromSearch("?view=training").panel).toBe("training");
      expect(parseNavFromSearch("?view=lens").panel).toBe("scientific-viewer");
    });

    it("parses Lens resource id(s)", () => {
      expect(parseNavFromSearch("?view=lens&resource=file-123")).toEqual(
        nav({ panel: "scientific-viewer", resourceFileIds: ["file-123"] })
      );
      expect(parseNavFromSearch("?view=lens&resource=a,b,c").resourceFileIds).toEqual(["a", "b", "c"]);
    });

    it("normalizes duplicate, padded and empty ids", () => {
      expect(parseNavFromSearch("?view=lens&resource=a,b,a").resourceFileIds).toEqual(["a", "b"]);
      expect(parseNavFromSearch("?view=lens&resource=%20a%20,,b,").resourceFileIds).toEqual(["a", "b"]);
      expect(parseNavFromSearch("?view=lens&resource=,").resourceFileIds).toEqual([]);
    });

    it("caps the id list at LENS_MAX_FILE_IDS", () => {
      const search = `?view=lens&resource=${manyIds(40).join(",")}`;
      expect(parseNavFromSearch(search).resourceFileIds).toEqual(manyIds(LENS_MAX_FILE_IDS));
    });

    it("keys a non-canonical URL identically to the state its restore produces", () => {
      // The Back-button invariant: a URL with duplicates or too many ids must parse
      // to a state whose key equals the key of the normalized request, so re-opening
      // that URL never looks like a new navigation.
      const raw = parseNavFromSearch(`?view=lens&resource=b,a,b,${manyIds(30).join(",")}`);
      const restored = nav({
        panel: "scientific-viewer",
        resourceFileIds: normalizeLensFileIds(["b", "a", "b", ...manyIds(30)]),
      });
      expect(navStateKey(raw)).toBe(navStateKey(restored));
      expect(raw.resourceFileIds).toHaveLength(LENS_MAX_FILE_IDS);
    });

    it("parses the Resources collection only on the resources panel", () => {
      expect(parseNavFromSearch("?view=resources&collection=col-42")).toEqual(
        nav({ panel: "resources", resourceCollectionId: "col-42" })
      );
      expect(parseNavFromSearch("?view=lens&collection=col-42").resourceCollectionId).toBeNull();
      expect(parseNavFromSearch("?collection=col-42").resourceCollectionId).toBeNull();
    });

    it("ignores a resource param outside Lens and tolerates unknown views", () => {
      expect(parseNavFromSearch("?view=resources&resource=file-1").resourceFileIds).toEqual([]);
      expect(parseNavFromSearch("?view=bogus").panel).toBe("chat");
    });

    it("round-trips through buildNavUrl", () => {
      const states: NavState[] = [
        nav({ panel: "chat" }),
        nav({ panel: "resources" }),
        nav({ panel: "resources", resourceCollectionId: "col-7" }),
        nav({ panel: "scientific-viewer", resourceFileIds: ["file-7"] }),
        nav({ panel: "scientific-viewer", resourceFileIds: ["x", "y"] }),
      ];
      for (const state of states) {
        const url = buildNavUrl(loc(""), state);
        const search = url.includes("?") ? url.slice(url.indexOf("?")) : "";
        expect(parseNavFromSearch(search)).toEqual(state);
      }
    });
  });

  describe("navStateKey", () => {
    it("is stable per panel and includes panel-specific identity", () => {
      expect(navStateKey(nav({ panel: "resources" }))).toBe("resources|");
      expect(navStateKey(nav({ panel: "resources", resourceCollectionId: "col-1" }))).toBe("resources|col-1");
      expect(navStateKey(nav({ panel: "scientific-viewer", resourceFileIds: ["a", "b"] }))).toBe(
        "scientific-viewer|a,b"
      );
      expect(navStateKey(nav({ panel: "scientific-viewer", resourceFileIds: ["a", "b"] }))).toBe(
        navStateKey(nav({ panel: "scientific-viewer", resourceFileIds: ["a", "b"] }))
      );
    });
  });
});

describe("lensDeepLinkFor", () => {
  it("builds the relative deep link for a single id", () => {
    expect(lensDeepLinkFor("file-123")).toBe("/?view=lens&resource=file-123");
    expect(lensDeepLinkFor(["file-123"])).toBe("/?view=lens&resource=file-123");
  });

  it("comma-joins multiple ids in one resource param, deduped and trimmed", () => {
    expect(lensDeepLinkFor(["a", " b ", "a"])).toBe("/?view=lens&resource=a,b");
  });

  it("caps at LENS_MAX_FILE_IDS like every other consumer", () => {
    const href = lensDeepLinkFor(manyIds(30));
    expect(href).toBe(`/?view=lens&resource=${manyIds(24).join(",")}`);
    expect(parseNavFromSearch(href.slice(href.indexOf("?"))).resourceFileIds).toEqual(manyIds(24));
  });

  it("percent-encodes each id individually", () => {
    expect(lensDeepLinkFor(["a b", "c&d"])).toBe("/?view=lens&resource=a%20b,c%26d");
  });

  it("omits the resource param when no usable id is given", () => {
    expect(lensDeepLinkFor([])).toBe("/?view=lens");
    expect(lensDeepLinkFor(["", "  "])).toBe("/?view=lens");
  });

  it("round-trips through parseNavFromSearch", () => {
    const href = lensDeepLinkFor(["x.1", "y:2"]);
    const nav = parseNavFromSearch(href.slice(href.indexOf("?")));
    expect(nav.panel).toBe("scientific-viewer");
    expect(nav.resourceFileIds).toEqual(["x.1", "y:2"]);
  });
});

describe("resolveLensLink", () => {
  const origin = "http://localhost:3000";

  it("accepts the relative form with and without the leading slash", () => {
    expect(resolveLensLink("/?view=lens&resource=file-1", origin)).toEqual({
      fileIds: ["file-1"],
      href: "/?view=lens&resource=file-1",
    });
    expect(resolveLensLink("?view=lens&resource=file-1", origin)).toEqual({
      fileIds: ["file-1"],
      href: "/?view=lens&resource=file-1",
    });
  });

  it("accepts multiple ids whether comma-joined raw or encoded", () => {
    expect(resolveLensLink("/?view=lens&resource=a,b", origin)?.fileIds).toEqual(["a", "b"]);
    expect(resolveLensLink("/?view=lens&resource=a%2Cb", origin)?.fileIds).toEqual(["a", "b"]);
    expect(resolveLensLink("/?view=lens&resource=a%2Cb", origin)?.href).toBe("/?view=lens&resource=a,b");
  });

  it("accepts the same-origin absolute form and preserves other params only as input", () => {
    expect(resolveLensLink("http://localhost:3000/?conversation=c9&view=lens&resource=file-2", origin)).toEqual({
      fileIds: ["file-2"],
      href: "/?view=lens&resource=file-2",
    });
  });

  it("accepts ultra://resource/<id>[/<name>] references", () => {
    expect(resolveLensLink("ultra://resource/file-3/cells%20stack.tif", origin)).toEqual({
      fileIds: ["file-3"],
      href: "/?view=lens&resource=file-3",
    });
    expect(resolveLensLink("ultra://resource/file-3", origin)?.fileIds).toEqual(["file-3"]);
    expect(resolveLensLink("ultra://resource/", origin)).toBeNull();
    expect(resolveLensLink("ultra://resource/bad%20id/x", origin)).toBeNull();
  });

  it("rejects foreign origins, including protocol-relative hrefs", () => {
    expect(resolveLensLink("https://evil.example/?view=lens&resource=file-1", origin)).toBeNull();
    expect(resolveLensLink("http://localhost:3001/?view=lens&resource=file-1", origin)).toBeNull();
    expect(resolveLensLink("//evil.example/?view=lens&resource=file-1", origin)).toBeNull();
    // A same-host https origin when the app runs on http is still foreign.
    expect(resolveLensLink("https://localhost:3000/?view=lens&resource=file-1", origin)).toBeNull();
  });

  it("never claims BisQue viewer, data_service or image_service shapes", () => {
    expect(
      resolveLensLink(
        "https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-abc",
        origin
      )
    ).toBeNull();
    expect(resolveLensLink("http://localhost:3000/data_service/00-abc", origin)).toBeNull();
    expect(resolveLensLink("http://localhost:3000/image_service/00-abc", origin)).toBeNull();
    expect(resolveLensLink("/client_service/view?resource=x&view=lens", origin)).toBeNull();
  });

  it("rejects missing, empty or malformed ids", () => {
    expect(resolveLensLink("/?view=lens", origin)).toBeNull();
    expect(resolveLensLink("/?view=lens&resource=", origin)).toBeNull();
    expect(resolveLensLink("/?view=lens&resource=,", origin)).toBeNull();
    expect(resolveLensLink("/?view=lens&resource=has%20space", origin)).toBeNull();
    expect(resolveLensLink("/?view=lens&resource=a/b", origin)).toBeNull();
    expect(resolveLensLink(`/?view=lens&resource=${"x".repeat(129)}`, origin)).toBeNull();
    // One bad id poisons the whole link rather than silently dropping it.
    expect(resolveLensLink("/?view=lens&resource=ok,bad%20id", origin)).toBeNull();
  });

  it("rejects non-Lens views, other paths and non-URLs", () => {
    expect(resolveLensLink("/?view=resources&collection=c1", origin)).toBeNull();
    expect(resolveLensLink("/?resource=file-1", origin)).toBeNull();
    expect(resolveLensLink("/app/?view=lens&resource=file-1", origin)).toBeNull();
    expect(resolveLensLink("", origin)).toBeNull();
    expect(resolveLensLink("not a url", origin)).toBeNull();
    expect(resolveLensLink("mailto:someone@example.com", origin)).toBeNull();
  });
});
