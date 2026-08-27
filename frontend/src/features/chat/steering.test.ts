/**
 * Mid-run steering — Phase 1 of "double texting".
 *
 * A steer reaches the RUNNING agent at its next model-call boundary. The
 * client contracts here protect three things: consent (steering is explicit
 * — ⌘Enter or the Steer button, never plain Enter), the Phase 0 settled
 * detection (the streaming assistant must stay the last message), and the
 * never-destroyed rule (a steered thought falls back to the queue or the
 * draft, it does not vanish).
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const appSource = readFileSync(path.join(process.cwd(), "src/App.tsx"), "utf8");
const apiSource = readFileSync(path.join(process.cwd(), "src/lib/api.ts"), "utf8");
const styles = readFileSync(path.join(process.cwd(), "src/styles.css"), "utf8");

const blockFrom = (start: string, end: string): string => {
  const startIndex = appSource.indexOf(start);
  expect(startIndex, `missing block: ${start.slice(0, 60)}`).toBeGreaterThan(-1);
  const endIndex = appSource.indexOf(end, startIndex);
  expect(endIndex, `unterminated block: ${start.slice(0, 60)}`).toBeGreaterThan(startIndex);
  return appSource.slice(startIndex, endIndex + end.length);
};

describe("entering a steer", () => {
  it("⌘Enter steers, in its own branch ABOVE the plain-Enter queue branch", () => {
    const steerBranch = appSource.indexOf("// ⌘Enter during a run STEERS");
    const queueBranch = appSource.indexOf("// Enter during a run queues the draft as a follow-up");
    expect(steerBranch).toBeGreaterThan(-1);
    expect(steerBranch).toBeLessThan(queueBranch);
    const branch = blockFrom("// ⌘Enter during a run STEERS", "steerFollowup();");
    expect(branch).toMatch(/\(event\.metaKey \|\| event\.ctrlKey\)/);
    expect(branch).toMatch(/!event\.nativeEvent\.isComposing/);
    expect(branch).toMatch(/activeSending/);
  });

  it("offers a visible Steer button left of Queue, with Stop still anchored last", () => {
    const running = blockFrom("{activeSending ? (", 'aria-label="Stop response"');
    const steer = running.indexOf('aria-label="Steer this run"');
    const queue = running.indexOf('aria-label="Queue follow-up"');
    expect(steer).toBeGreaterThan(-1);
    expect(steer).toBeLessThan(queue);
    expect(queue).toBeLessThan(running.indexOf('aria-label="Stop response"'));
  });

  it("falls back to queueing when the run id has not streamed back yet", () => {
    const steer = blockFrom("const steerFollowup = useCallback", "uploadPendingFiles,\n  ]);");
    expect(steer).toMatch(/if \(!runId\)/);
  });

  it("honours the same refusal states as the send and queue paths", () => {
    const steer = blockFrom("const steerFollowup = useCallback", "makeId()");
    expect(steer).toMatch(/slashMenuOpen\s+\|\|\s+composerResourcePickerOpen\s+\|\|\s+composerNotePickerOpen/);
  });

  it("queues Note-bearing follow-ups as a newly scoped run", () => {
    const steer = blockFrom("const steerFollowup = useCallback", "makeId()");
    expect(steer).toMatch(
      /conversation\.selectedNotes\.length > 0 \|\|\s*notesSearchScopeState\(\s*text,\s*excludedNoteIntentText,\s*conversation\.noteSearchScopeOverride\s*\)\.active/
    );
    expect(steer).toMatch(/queueFollowup\(\);\s*return;/);
  });
});

describe("the optimistic steering message", () => {
  it("inserts BEFORE the streaming assistant row — Phase 0's settled detection needs the assistant last", () => {
    const steer = blockFrom("const steerFollowup = useCallback", "void (async () => {");
    const insert = steer.indexOf("messages.findIndex((item) => item.id === current.streamingMessageId)");
    const splice = steer.indexOf("messages.splice(insertAt, 0, message)");
    expect(insert).toBeGreaterThan(-1);
    expect(splice).toBeGreaterThan(insert);
  });

  it("adopts the durable transcript row id on success so hydration converges", () => {
    expect(appSource).toMatch(/id: record\.message_id \|\| item\.id/);
  });

  it("is removed on failure — and the text is never destroyed", () => {
    const steer = blockFrom(
      "const steerFollowup = useCallback",
      "uploadPendingFiles,\n  ]);"
    );
    const catchAt = steer.indexOf(".catch(async (error: unknown) => {");
    expect(catchAt).toBeGreaterThan(-1);
    const failure = steer.slice(catchAt);
    // A lost RESPONSE is not a lost steer: verify server-side before
    // restoring the draft, or a re-send duplicates the transcript.
    expect(failure).toMatch(/listRunSteerMessages\(runId\)/);
    const verifyAt = failure.indexOf("listRunSteerMessages");
    const restoreAt = failure.indexOf("filter((item) => item.steerId !== steerId)");
    expect(verifyAt).toBeGreaterThan(-1);
    expect(verifyAt).toBeLessThan(restoreAt);
    expect(failure).toMatch(/filter\(\(item\) => item\.steerId !== steerId\)/);
    // steering_closed → Phase 0 queue with the growing-message rule.
    expect(failure).toMatch(/isSteeringClosedError\(error\)/);
    expect(failure).toMatch(/queuedFollowup: current\.queuedFollowup/);
    // any other failure → back to the draft with the append rule.
    expect(failure).toMatch(/previous\.trim\(\) \? `\$\{previous\.replace/);
  });
});

describe("steer lifecycle events", () => {
  it("routes steer.* run events away from the assistant fold in BOTH stream consumers", () => {
    const matches = appSource.match(/if \(applySteerRunEvent\((?:target\.conversationId|conversationId), runEvent\)\) \{\s*return;/g);
    expect(matches?.length).toBe(2);
  });

  it("routes steer.* events in the no-stream poll fallback too — the eyebrow must not stick at pending", () => {
    // The 1250ms poll is the ONLY consumer when no SSE stream is registered.
    expect(appSource).toMatch(/if \(applySteerRunEvent\(conversationId, event\)\) \{\s*continue;/);
  });

  it("Retry/Edit resolves the ORIGINATING prompt, never the steer, and folds the turn's steers in", () => {
    const retry = blockFrom(
      "const retryAssistantResponse = useCallback",
      "})().finally(() => noteTurnResealInFlightRef.current.delete(resealKey));"
    );
    expect(retry).toMatch(/if \(candidate\.steering\) \{\s*turnSteerTexts\.unshift\(candidate\.content\);/);
    expect(retry).toMatch(/\[originatingUserMessage\.content, \.\.\.turnSteerTexts\]/);
    expect(retry).toMatch(/originatingUserMessage\.noteReferences/);
    expect(retry).toMatch(/originatingUserMessage\.excludedNoteIntentText/);
    expect(retry).toMatch(/candidate\.excludedNoteIntentText/);
    expect(retry).toMatch(/selectedNotes: noteReferences/);
    expect(retry).toMatch(/excludedNoteIntentText: historicalExcludedNoteIntentText/);
    expect(retry).toMatch(/activeSelectionContext: noteSelectionContextForChips/);
    expect(retry).toMatch(/await resealTurnNotes\(conversationId, historicalReferences\)/);
    expect(retry.indexOf("await resealTurnNotes")).toBeLessThan(
      retry.indexOf("removeMessageWithPairedResponse")
    );
    // The removal takes the turn's steering bubbles with it — no orphans.
    expect(appSource).toMatch(/candidate\.role === "user" && candidate\.steering/);
  });

  it("only moves the lifecycle forward — a replayed steer.received cannot demote applied", () => {
    const fold = blockFrom("const applySteerRunEvent = useCallback", "return true;\n    },\n    [updateConversation]");
    expect(fold).toMatch(/existing\.steering && existing\.steering !== "pending"/);
  });

  it("materializes a cross-tab steer without ever putting a user message after the last assistant", () => {
    const fold = blockFrom("const applySteerRunEvent = useCallback", "return true;\n    },\n    [updateConversation]");
    expect(fold).toContain('messages[index].role === "assistant"');
    expect(fold).toContain("messages.splice(insertAt, 0, message)");
  });

  it("is defined above every stream consumer that closes over it (the TDZ trap)", () => {
    const definition = appSource.indexOf("const applySteerRunEvent = useCallback");
    const firstUse = appSource.indexOf("applySteerRunEvent(target.conversationId");
    expect(definition).toBeGreaterThan(-1);
    expect(definition).toBeLessThan(firstUse);
  });
});

describe("durability and hydration", () => {
  it("persists steering identity in the snapshot round-trip", () => {
    expect(appSource).toMatch(/steering: message\.steering,\s*steerId: message\.steerId,/);
    expect(appSource).toMatch(/row\.steering === "pending" \|\|\s*row\.steering === "applied"/);
  });

  it("persists pasted-reference provenance so retry cannot mint Notes authority", () => {
    expect(appSource).toMatch(/excludedNoteIntentText: message\.excludedNoteIntentText \?\? \[\]/);
    expect(appSource).toMatch(/excludedNoteIntentText: Array\.isArray\(row\.excludedNoteIntentText\)/);
    expect(appSource).toMatch(/excludedNoteIntentText:\s*excludedNoteIntentText\.length > 0/);
  });

  it("hydrates control-plane steering rows as historic steering messages", () => {
    expect(apiSource).toMatch(/asOptionalString\(metadata\.kind\) === "steering"/);
    expect(apiSource).toMatch(/state\.steering = "historic";/);
  });

  it("detects the steering_closed 409 by code, not by status alone", () => {
    expect(apiSource).toMatch(/error\.status === 409/);
    expect(apiSource).toMatch(/code === "steering_closed"/);
  });
});

describe("the steering eyebrow", () => {
  it("states each lifecycle phase in words, with a live region for the transition", () => {
    expect(appSource).toContain("Steering — will be seen shortly");
    expect(appSource).toContain("Seen by the agent");
    expect(appSource).toContain("Run ended before this was read — carries into the next turn");
    expect(appSource).toContain("Steered mid-run");
    const eyebrow = blockFrom('className="chat-steering-eyebrow"', "</span>");
    expect(eyebrow).toContain('aria-live="polite"');
  });

  it("keeps the label quieter than the content, reserving color for missed", () => {
    expect(styles).toMatch(/\.chat-steering-eyebrow \{[^}]*font-size: 0\.72rem/s);
    expect(styles).toMatch(/\.chat-steering-eyebrow\[data-steering="missed"\]/);
    const reduced = styles.indexOf(".chat-steering-eyebrow", styles.indexOf("@media (prefers-reduced-motion: reduce)", styles.indexOf(".chat-steering-eyebrow")));
    expect(reduced).toBeGreaterThan(-1);
  });
});

describe("steer attachments", () => {
  it("uploads the pending files FIRST and stamps their ids onto the steer payload", () => {
    const flow = blockFrom("const pendingFilesSnapshot = conversation.pendingFiles;", ".steerRun(runId, { steerId, text, fileIds })");
    expect(flow).toContain("await uploadPendingFiles(");
    expect(flow).toContain("uploadResult.newlyUploadedFiles.map((file) => file.file_id)");
    // Upload strictly precedes the steer POST.
    expect(flow.indexOf("await uploadPendingFiles(")).toBeLessThan(
      flow.indexOf(".steerRun(runId, { steerId, text, fileIds })")
    );
  });

  it("sends file_ids in the steer request body — text alone can never carry authority", () => {
    const steerClient = apiSource.slice(
      apiSource.indexOf("async steerRun("),
      apiSource.indexOf("listRunSteerMessages")
    );
    expect(steerClient).toContain("file_ids: input.fileIds ?? []");
  });

  it("withdraws the optimistic row and returns the draft when the upload fails — no partial steer", () => {
    const failure = blockFrom("// Never send a partial steer", "return;");
    expect(failure).toContain("current.messages.filter((item) => item.steerId !== steerId)");
    expect(failure).toContain("setActivePromptValue");
    expect(failure).toContain("Attachment upload failed");
  });

  it("shows the attached file names on the optimistic steering message", () => {
    const optimistic = blockFrom("id: `steer-local-${steerId}`", "steerId,");
    const message = blockFrom("id: `steer-local-${steerId}`", "uploadedFileNames.length > 0");
    expect(optimistic).toBeTruthy();
    expect(message).toContain("uploadedFileNames:");
  });
});
