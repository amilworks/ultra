import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

/* Source-level pins for the composer's app-side wiring: the pieces whose
   correctness is a matter of ordering or gating rather than of any one
   function. The editor and the grammar carry their own unit tests. */
const read = (relativePath: string): string =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8");

const appSource = read("src/App.tsx");
const composerSource = read("src/components/composer/Composer.tsx");
const editorSource = read("src/components/composer/ComposerEditor.tsx");

describe("composer contract", () => {
  it("mounts one composer, behind the app's upload funnel, with the app as its adapter", () => {
    expect(appSource.match(/<Composer\n/g)?.length).toBe(1);
    expect(appSource).toMatch(/<FileUpload[\s\S]{0,800}<Composer\n/);
    expect(appSource).not.toMatch(/PromptInputTextarea|BriefOverlay|measureTextareaCaret/);
  });

  it("loads the editor as its own chunk and covers the first paint with the fallback", () => {
    expect(composerSource).toMatch(/export const loadComposerEditorModule = \(\) => import\("\.\/ComposerEditor"\);/);
    expect(composerSource).toMatch(/<Suspense fallback=\{<ComposerFallbackEditor ref=\{fallbackRef\} \{\.\.\.editorProps\} \/>\}>/);
    expect(composerSource).toMatch(/if \(focusedRef\.current\) \{\s*editorRef\.current\?\.focus\(\{ caret: "end", preventScroll: true \}\);/);
  });

  it("keeps the text as the source of truth on both sides", () => {
    expect(appSource).toMatch(/value=\{activePrompt\}/);
    expect(appSource).toMatch(/onValueChange=\{setActivePromptValue\}/);
    expect(editorSource).toMatch(/if \(current !== props\.value \|\| tokensDrift\(view\.state\.doc, props\.value, props\.tokens\)\)/);
  });

  it("lets the editor's tokens drive the registry and the staging", () => {
    expect(appSource).toMatch(/onTokensChange=\{handleComposerTokensChange\}/);
    expect(appSource).toMatch(/if \(dropped\.length > 0\) \{\s*unstageBriefFiles\(dropped\);/);
    // Files that arrive by any other path get a token, idempotently.
    expect(appSource).toMatch(/composerRef\.current\?\.appendToken\(token\);/);
    expect(editorSource).toMatch(/if \(!view \|\| findTokenPosition\(view\.state\.doc, token\.fileId\) !== null\) \{\s*return;/);
  });

  it("judges no token gone before the conversation hydrates, and holds the send when one is", () => {
    expect(appSource).toMatch(
      /activeConversationHydrated\s*\?\s*activeBriefRegistry\s*\.filter\(\(token\) => !briefAvailableFileIds\.has\(token\.fileId\)\)/
    );
    expect(appSource).toMatch(/slashMenuOpen \|\|\s*briefGoneFileIds\.length > 0;/);
    expect(appSource).toMatch(/if \(briefGoneFileIds\.length > 0\) \{\s*showErrorToast\(/);
    expect(appSource).toMatch(/action: \{ label: "choose another", onClick: replaceFirstGoneBriefToken \}/);
    expect(appSource).toMatch(/composerRef\.current\?\.reopenMentionFor\(gone\);/);
  });

  it("says what a file-first workflow needs in the whisper instead of a badge row", () => {
    expect(appSource).toMatch(/briefWorkflowNeedsFiles && briefWorkflowLabel/);
    expect(appSource).toMatch(/label: "choose from your library"/);
    expect(appSource).not.toMatch(/data-testid="composer-workflow-chip"/);
    expect(composerSource).toMatch(/data-testid="composer-workflow-chip"/);
  });

  it("keys: the composer owns Enter and the @ picker, the app keeps the slash menu and recall", () => {
    expect(composerSource).toMatch(/if \(open && activeMention && !event\.isComposing\) \{/);
    // Enter never sends past an open picker that is still answering.
    expect(composerSource).toMatch(/if \(activeMention\.loading\) \{\s*event\.preventDefault\(\);\s*return true;/);
    expect(appSource).toMatch(/setBriefMentionLoading\(true\);\s*const timer = window\.setTimeout/);
    expect(composerSource).toMatch(/return state\.onKeyDown\?\.\(event\) \?\? false;/);
    expect(appSource).toMatch(/onKeyDown=\{handleComposerKeyDown\}/);
    expect(appSource).toMatch(/const handleComposerKeyDown = \(event: KeyboardEvent\): boolean => \{/);
  });

  it("only the mirrored text of a token is decoration; the chips are real controls", () => {
    expect(editorSource).toMatch(/this\.dom\.setAttribute\("contenteditable", "false"\);/);
    expect(editorSource).toMatch(/this\.remove\.setAttribute\("aria-label", `Remove \$\{label\}`\);/);
    expect(composerSource).toMatch(/<span className="composer-prefix">/);
    expect(composerSource).not.toMatch(/aria-hidden="true">\s*<span className="composer-prefix"/);
  });
});
