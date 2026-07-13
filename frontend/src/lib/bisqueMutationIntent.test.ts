import { describe, expect, it } from "vitest";

import { remoteMutationIntentsForUserText } from "./bisqueMutationIntent";

describe("remoteMutationIntentsForUserText", () => {
  it("grants only explicit current-turn BisQue mutation requests", () => {
    expect(remoteMutationIntentsForUserText("Please upload the resulting plot to BisQue.")).toEqual([
      "bisque.upload",
    ]);
    expect(
      remoteMutationIntentsForUserText(
        "Push the outputs to my linked BisQue account and create a BisQue dataset grouping them."
      )
    ).toEqual(["bisque.upload", "bisque.create_dataset"]);
  });

  it("rejects explanatory, hypothetical, and negated text", () => {
    for (const text of [
      "Explain how users upload files to BisQue.",
      "Can Ultra upload files to BisQue?",
      "If I asked you to upload the report to BisQue, what would happen?",
      "I didn't ask you to upload anything to BisQue.",
      "Do not create a BisQue dataset.",
      "Create a local dataset from the results.",
    ]) {
      expect(remoteMutationIntentsForUserText(text)).toEqual([]);
    }
  });
});
