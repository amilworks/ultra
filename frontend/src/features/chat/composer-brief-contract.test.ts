import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/* Source-level pins for the brief composer: the pieces whose correctness is a
   matter of ordering or gating in App.tsx rather than of any one function. */
const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const overlaySource = readFileSync(
  path.join(process.cwd(), "src/components/chat/BriefOverlay.tsx"),
  "utf8"
);

describe("brief composer contract", () => {
  it("mounts the overlay after the textarea so its ref exists when the overlay measures", () => {
    expect(appSource).toMatch(
      /<PromptInputTextarea[\s\S]*?\n {20}\/>\n\s*\{activeConversationHydrated \? \(\s*<BriefOverlay/
    );
  });

  it("re-measures the overlay when the textarea attaches a frame late", () => {
    expect(overlaySource).toMatch(/requestAnimationFrame\(\(\) => setMountTick/);
    expect(overlaySource).toMatch(/\}, \[textareaRef, text, syncKey, mountTick\]\);/);
  });

  it("hides only the mirrored text from assistive tech, never the prefix controls", () => {
    expect(overlaySource).not.toMatch(/className=\{cn\("brief-overlay", className\)\} style=\{box\} aria-hidden/);
    expect(overlaySource).toMatch(/className="brief-overlay-mirror" aria-hidden="true"/);
  });

  it("judges no token gone before the conversation hydrates", () => {
    expect(appSource).toMatch(
      /activeConversationHydrated\s*\?\s*activeBriefRegistry\s*\.filter\(\(token\) => !briefAvailableFileIds\.has\(token\.fileId\)\)/
    );
  });

  it("says what a file-first workflow needs in the whisper instead of a second badge row", () => {
    expect(appSource).toMatch(/briefWorkflowNeedsFiles \? \([\s\S]{0,600}choose from your library/);
    expect(appSource).not.toMatch(/data-testid="composer-workflow-chip"/);
    expect(appSource).not.toMatch(/Clear workflow/);
  });

  it("recovers a pill for a draft that already says @label instead of inserting a second one", () => {
    expect(appSource).toMatch(/if \(briefFileTokensInText\(text, \[token\]\)\.length > 0\) \{\s*continue;/);
  });

  it("blocks the send while a tokenized file is gone from the library", () => {
    expect(appSource).toMatch(/slashMenuOpen \|\|\s*briefGoneFileIds\.length > 0;/);
    expect(appSource).toMatch(/if \(briefGoneFileIds\.length > 0\) \{\s*showErrorToast\(/);
  });
});
