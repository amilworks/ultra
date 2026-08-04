import { describe, expect, it } from "vitest";

import { buildNavUrl, navStateKey, parseNavFromSearch, type NavState } from "./navUrl";

const loc = (search: string, pathname = "/", hash = "") => ({ pathname, search, hash });

const nav = (partial: Partial<NavState> & Pick<NavState, "panel">): NavState => ({
  resourceFileIds: [],
  resourceCollectionId: null,
  ...partial,
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
