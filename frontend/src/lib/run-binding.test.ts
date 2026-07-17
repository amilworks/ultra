import { describe, expect, it } from "vitest";

import type { ChatRequest } from "@/types";
import { ApiClient } from "./api";

type RunRequestBuilder = {
  buildV2RunRequest: (request: ChatRequest) => Record<string, unknown>;
};

const buildWirePayload = (overrides: Partial<ChatRequest> = {}): Record<string, unknown> => {
  const request: ChatRequest = {
    messages: [{ role: "user", content: "Upload this sensor series to BisQue." }],
    uploaded_files: [],
    ...overrides,
  };
  const client = new ApiClient({ baseUrl: "https://ultra.example.org" });
  return (client as unknown as RunRequestBuilder).buildV2RunRequest(request);
};

describe("run binding", () => {
  it("binds explicit remote-mutation authority into the run envelope", () => {
    expect(
      buildWirePayload({ remote_mutation_intents: ["bisque.upload"] })
    ).toMatchObject({ remote_mutation_intents: ["bisque.upload"] });
  });
});
