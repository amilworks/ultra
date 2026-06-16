import { describe, expect, it } from "vitest";

import { buildNavUrl, navStateKey, parseNavFromSearch } from "./navUrl";

const loc = (search: string, pathname = "/", hash = "") => ({ pathname, search, hash });

describe("navUrl", () => {
  describe("buildNavUrl", () => {
    it("encodes chat as the absence of a view param", () => {
      expect(buildNavUrl(loc(""), { panel: "chat", resourceFileIds: [] })).toBe("/");
      // Clears a stale view/resource when returning to chat.
      expect(buildNavUrl(loc("?view=lens&resource=abc"), { panel: "chat", resourceFileIds: [] })).toBe("/");
    });

    it("encodes the resources / admin / training panels", () => {
      expect(buildNavUrl(loc(""), { panel: "resources", resourceFileIds: [] })).toBe("/?view=resources");
      expect(buildNavUrl(loc(""), { panel: "admin", resourceFileIds: [] })).toBe("/?view=admin");
      expect(buildNavUrl(loc(""), { panel: "training", resourceFileIds: [] })).toBe("/?view=training");
    });

    it("encodes Lens with its resource id(s)", () => {
      expect(buildNavUrl(loc(""), { panel: "scientific-viewer", resourceFileIds: ["file-123"] })).toBe(
        "/?view=lens&resource=file-123"
      );
      expect(
        buildNavUrl(loc(""), { panel: "scientific-viewer", resourceFileIds: ["a", "b"] })
      ).toBe("/?view=lens&resource=a%2Cb");
    });

    it("omits the resource param when there are no file ids", () => {
      expect(buildNavUrl(loc(""), { panel: "scientific-viewer", resourceFileIds: [] })).toBe("/?view=lens");
    });

    it("PRESERVES other query params (e.g. conversation) so the two URL layers never clobber", () => {
      expect(
        buildNavUrl(loc("?conversation=conv-9"), { panel: "scientific-viewer", resourceFileIds: ["file-1"] })
      ).toBe("/?conversation=conv-9&view=lens&resource=file-1");
      // Switching to chat keeps the conversation but drops view/resource.
      expect(
        buildNavUrl(loc("?conversation=conv-9&view=lens&resource=file-1"), { panel: "chat", resourceFileIds: [] })
      ).toBe("/?conversation=conv-9");
    });

    it("preserves pathname and hash", () => {
      expect(buildNavUrl(loc("", "/code", "#section"), { panel: "resources", resourceFileIds: [] })).toBe(
        "/code?view=resources#section"
      );
    });
  });

  describe("parseNavFromSearch", () => {
    it("defaults to chat when no view param", () => {
      expect(parseNavFromSearch("")).toEqual({ panel: "chat", resourceFileIds: [] });
      expect(parseNavFromSearch("?conversation=conv-1")).toEqual({ panel: "chat", resourceFileIds: [] });
    });

    it("maps view -> panel", () => {
      expect(parseNavFromSearch("?view=resources").panel).toBe("resources");
      expect(parseNavFromSearch("?view=admin").panel).toBe("admin");
      expect(parseNavFromSearch("?view=training").panel).toBe("training");
      expect(parseNavFromSearch("?view=lens").panel).toBe("scientific-viewer");
    });

    it("parses Lens resource id(s)", () => {
      expect(parseNavFromSearch("?view=lens&resource=file-123")).toEqual({
        panel: "scientific-viewer",
        resourceFileIds: ["file-123"],
      });
      expect(parseNavFromSearch("?view=lens&resource=a,b,c").resourceFileIds).toEqual(["a", "b", "c"]);
    });

    it("ignores a resource param outside Lens and tolerates unknown views", () => {
      expect(parseNavFromSearch("?view=resources&resource=file-1").resourceFileIds).toEqual([]);
      expect(parseNavFromSearch("?view=bogus").panel).toBe("chat");
    });

    it("round-trips through buildNavUrl", () => {
      const states = [
        { panel: "chat" as const, resourceFileIds: [] },
        { panel: "resources" as const, resourceFileIds: [] },
        { panel: "scientific-viewer" as const, resourceFileIds: ["file-7"] },
        { panel: "scientific-viewer" as const, resourceFileIds: ["x", "y"] },
      ];
      for (const state of states) {
        const url = buildNavUrl(loc(""), state);
        const search = url.includes("?") ? url.slice(url.indexOf("?")) : "";
        expect(parseNavFromSearch(search)).toEqual(state);
      }
    });
  });

  describe("navStateKey", () => {
    it("is stable per panel and includes the resource list only for Lens", () => {
      expect(navStateKey({ panel: "resources", resourceFileIds: [] })).toBe("resources");
      expect(navStateKey({ panel: "scientific-viewer", resourceFileIds: ["a", "b"] })).toBe("scientific-viewer|a,b");
      expect(navStateKey({ panel: "scientific-viewer", resourceFileIds: ["a", "b"] })).toBe(
        navStateKey({ panel: "scientific-viewer", resourceFileIds: ["a", "b"] })
      );
    });
  });
});
