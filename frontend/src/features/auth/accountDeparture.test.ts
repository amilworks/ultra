import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";
import { requestAuthenticatedAccountDeparture } from "./accountDeparture";

describe("authenticated account departure", () => {
  it.each(["logout", "unlink"] as const)(
    "does not turn a failed %s request into local success",
    async (departure) => {
      const failure = new Error(`${departure} unavailable`);
      const apiClient = {
        logoutBisque: vi.fn().mockRejectedValue(failure),
        unlinkBisqueAccount: vi.fn().mockRejectedValue(failure),
      } as unknown as ApiClient;

      await expect(
        requestAuthenticatedAccountDeparture(apiClient, departure)
      ).rejects.toBe(failure);
      expect(apiClient.logoutBisque).toHaveBeenCalledTimes(departure === "logout" ? 1 : 0);
      expect(apiClient.unlinkBisqueAccount).toHaveBeenCalledTimes(
        departure === "unlink" ? 1 : 0
      );
    }
  );

  it("returns a logout redirect only after the server confirms it", async () => {
    const apiClient = {
      logoutBisque: vi.fn().mockResolvedValue({ logout_url: " /signed-out " }),
      unlinkBisqueAccount: vi.fn(),
    } as unknown as ApiClient;

    await expect(
      requestAuthenticatedAccountDeparture(apiClient, "logout")
    ).resolves.toBe("/signed-out");
  });
});
