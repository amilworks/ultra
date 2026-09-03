import { EditorState, TextSelection } from "prosemirror-state";
import { describe, expect, it } from "vitest";

import {
  appendTokenAt,
  deleteTokenBackward,
  deleteTokenForward,
  docFromText,
  insertTokenAt,
  isDocEmpty,
  mentionAtSelection,
  removeTokenNode,
  reopenMentionAt,
  replaceDocFromText,
  textFromDoc,
  tokensInDoc,
} from "./composerSchema";

const registry = [
  { label: "scan.tif", fileId: "f1" },
  { label: "scan.tif (2)", fileId: "f2" },
];

const stateFor = (text: string, caret?: number) => {
  const doc = docFromText(text, registry);
  let state = EditorState.create({ doc });
  if (caret !== undefined) {
    state = state.apply(state.tr.setSelection(TextSelection.create(doc, caret)));
  }
  return state;
};

describe("docFromText / textFromDoc", () => {
  it("round-trips lines and tokens exactly", () => {
    const text = "Register @scan.tif against @scan.tif (2)\nthen compare";
    const doc = docFromText(text, registry);
    expect(doc.childCount).toBe(2);
    expect(textFromDoc(doc)).toBe(text);
    expect(tokensInDoc(doc)).toEqual(registry);
  });
  it("leaves an unregistered @word as plain text", () => {
    const doc = docFromText("mail user@example.com now", registry);
    expect(tokensInDoc(doc)).toEqual([]);
    expect(textFromDoc(doc)).toBe("mail user@example.com now");
  });
  it("knows an empty document", () => {
    expect(isDocEmpty(docFromText("", registry))).toBe(true);
    expect(isDocEmpty(docFromText("x", registry))).toBe(false);
    expect(isDocEmpty(docFromText("\n", registry))).toBe(false);
  });
});

describe("mentionAtSelection", () => {
  it("finds the @query ending at the caret", () => {
    const state = stateFor("Register @sc", 13);
    expect(mentionAtSelection(state)).toEqual({ from: 10, to: 13, query: "sc" });
  });
  it("accepts a bare @ and an @ after opening punctuation", () => {
    expect(mentionAtSelection(stateFor("look (@", 8))?.query).toBe("");
  });
  it("ignores a glued @, a token, a selection, and a space inside the query", () => {
    expect(mentionAtSelection(stateFor("user@exa", 9))).toBeNull();
    expect(mentionAtSelection(stateFor("x @scan.tif", 4))).toBeNull();
    expect(mentionAtSelection(stateFor("@sc x", 6))).toBeNull();
    const doc = docFromText("Register @sc", registry);
    const ranged = EditorState.create({ doc }).apply(
      EditorState.create({ doc }).tr.setSelection(TextSelection.create(doc, 11, 13))
    );
    expect(mentionAtSelection(ranged)).toBeNull();
  });
});

describe("insertTokenAt", () => {
  it("replaces the mention run, pads a space, and leaves the caret after it", () => {
    const state = stateFor("Register @sc", 13);
    const mention = mentionAtSelection(state)!;
    const next = state.apply(insertTokenAt(state, mention.from, mention.to, registry[0]));
    expect(textFromDoc(next.doc)).toBe("Register @scan.tif ");
    expect(next.selection.from).toBe(next.doc.content.size - 1);
  });
  it("reuses an existing following space and pads before a glued word", () => {
    const state = stateFor("Register @sc here", 13);
    const mention = mentionAtSelection(state)!;
    const next = state.apply(insertTokenAt(state, mention.from, mention.to, registry[0]));
    expect(textFromDoc(next.doc)).toBe("Register @scan.tif here");
    const glued = stateFor("Register", 9);
    const padded = glued.apply(insertTokenAt(glued, 9, 9, registry[0]));
    expect(textFromDoc(padded.doc)).toBe("Register @scan.tif ");
  });
});

describe("appendTokenAt", () => {
  it("lands at the caret when focused, at the end otherwise", () => {
    const state = stateFor("one two", 4);
    expect(textFromDoc(state.apply(appendTokenAt(state, registry[0], true)).doc)).toBe("one @scan.tif two");
    expect(textFromDoc(state.apply(appendTokenAt(state, registry[0], false)).doc)).toBe("one two @scan.tif ");
  });
});

describe("removeTokenNode / reopenMentionAt", () => {
  it("removes the token and one padding space", () => {
    const state = stateFor("a @scan.tif b");
    expect(textFromDoc(state.apply(removeTokenNode(state, "f1")!).doc)).toBe("a b");
    const trailing = stateFor("a @scan.tif");
    expect(textFromDoc(trailing.apply(removeTokenNode(trailing, "f1")!).doc)).toBe("a");
    const padded = stateFor("a @scan.tif ");
    expect(textFromDoc(padded.apply(removeTokenNode(padded, "f1")!).doc)).toBe("a");
    expect(removeTokenNode(state, "missing")).toBeNull();
  });
  it("turns a token back into an active @ mention", () => {
    const state = stateFor("a @scan.tif b");
    const next = state.apply(reopenMentionAt(state, "f1")!);
    expect(textFromDoc(next.doc)).toBe("a @ b");
    expect(mentionAtSelection(next)).toEqual({ from: 3, to: 4, query: "" });
  });
});

describe("replaceDocFromText", () => {
  it("rebuilds from text and keeps the caret in range", () => {
    const state = stateFor("hello world", 6);
    const next = state.apply(replaceDocFromText(state, "hi @scan.tif", registry));
    expect(textFromDoc(next.doc)).toBe("hi @scan.tif");
    expect(next.selection.from).toBeLessThanOrEqual(next.doc.content.size - 1);
    const atEnd = state.apply(replaceDocFromText(state, "short", registry, "end"));
    expect(atEnd.selection.from).toBe(atEnd.doc.content.size - 1);
  });
});

describe("deleteTokenBackward / deleteTokenForward", () => {
  it("removes the token in one keystroke from either side, and stays out of ordinary text", () => {
    const afterToken = stateFor("a @scan.tif b", 4);
    expect(textFromDoc(afterToken.apply(deleteTokenBackward(afterToken)!).doc)).toBe("a  b");
    const beforeToken = stateFor("a @scan.tif b", 3);
    expect(textFromDoc(beforeToken.apply(deleteTokenForward(beforeToken)!).doc)).toBe("a  b");
    expect(deleteTokenBackward(stateFor("plain text", 5))).toBeNull();
    expect(deleteTokenForward(stateFor("plain text", 5))).toBeNull();
    expect(deleteTokenBackward(stateFor("a @scan.tif b", 5))).toBeNull();
  });
});
