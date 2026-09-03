import { describe, expect, it } from "vitest";

import {
  clampMentionAnchor,
  composerKeysHint,
  composerPlaceholder,
  deriveComposerStage,
} from "./composerModel";

const quiet = {
  running: false,
  focused: false,
  hasText: false,
  hasTokens: false,
  hasFiles: false,
  hasWorkflow: false,
  menuOpen: false,
  welcomeStage: false,
};

describe("deriveComposerStage", () => {
  it("rests only when nothing is happening", () => {
    expect(deriveComposerStage(quiet)).toBe("rest");
  });
  it("composes on focus, content, files, a workflow, an open menu, or the welcome stage", () => {
    for (const key of ["focused", "hasText", "hasTokens", "hasFiles", "hasWorkflow", "menuOpen", "welcomeStage"] as const) {
      expect(deriveComposerStage({ ...quiet, [key]: true })).toBe("composing");
    }
  });
  it("running outranks everything", () => {
    expect(deriveComposerStage({ ...quiet, running: true, focused: true })).toBe("running");
  });
});

describe("composerKeysHint", () => {
  it("states the run contract: ↵ queues, ⌘↵ steers", () => {
    expect(composerKeysHint(true)).toBe("queue for after · ⌘↵ steer");
    expect(composerKeysHint(false)).toBe("send · ⇧↵ new line");
  });
});

describe("composerPlaceholder", () => {
  const base = { hydrated: true, welcomeStage: false, readMode: false, running: false, hasTokens: false, hasFiles: false, phone: false };
  it("teaches the grammar once working, and keeps the welcome invitation on the welcome stage", () => {
    expect(composerPlaceholder(base)).toBe("Ask Ultra — @ to bring in a file, / for a workflow");
    // Phones drop the cues: the bar's status box cannot hold them untruncated.
    expect(composerPlaceholder({ ...base, phone: true })).toBe("Ask Ultra");
    expect(composerPlaceholder({ ...base, welcomeStage: true })).toBe("Describe a question, dataset, or experiment…");
  });
  it("names what still works while collapsed, and what a run accepts", () => {
    expect(composerPlaceholder({ ...base, readMode: true })).toBe("Just start typing");
    expect(composerPlaceholder({ ...base, running: true, readMode: true })).toBe("Steer this run, or queue for after");
  });
  it("goes quiet once a file is in the brief, and says so before hydration", () => {
    expect(composerPlaceholder({ ...base, hasTokens: true })).toBe("Ask Ultra");
    expect(composerPlaceholder({ ...base, hydrated: false })).toBe("Loading chat…");
  });
});

describe("clampMentionAnchor", () => {
  it("keeps the picker inside the surface", () => {
    expect(clampMentionAnchor(900, 1000)).toBe(620);
    expect(clampMentionAnchor(-4, 1000)).toBe(0);
    expect(clampMentionAnchor(100, 300)).toBe(0);
  });
});
