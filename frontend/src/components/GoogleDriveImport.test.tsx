/// <reference types="node" />

import { readFileSync } from "node:fs";
import path from "node:path";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient, GoogleDriveStatus } from "@/lib/api";

import {
  GoogleDriveImportFlow,
  type PickedDriveFile,
  type PickerOpener,
} from "./GoogleDriveImport";

const componentSource = readFileSync(
  path.join(process.cwd(), "src/components/GoogleDriveImport.tsx"),
  "utf8"
);

type Deferred<T> = { promise: Promise<T>; resolve: (value: T) => void; reject: (error: Error) => void };

const deferred = <T,>(): Deferred<T> => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const CONNECTED: GoogleDriveStatus = {
  enabled: true,
  connected: true,
  account_email: "scientist@ucsb.edu",
  status: "active",
};

const TOKEN = {
  access_token: "picker-access-token",
  expires_at: new Date(Date.now() + 3600e3).toISOString(),
  picker_api_key: "",
  app_id: "",
};

type ClientOverrides = Partial<{
  googleDriveStatus: ReturnType<typeof vi.fn>;
  googleDriveAuthorizeURL: ReturnType<typeof vi.fn>;
  googleDrivePickerToken: ReturnType<typeof vi.fn>;
  googleDriveImportFile: ReturnType<typeof vi.fn>;
}>;

const fakeClient = (overrides: ClientOverrides = {}) => {
  const client = {
    googleDriveStatus: vi.fn().mockResolvedValue(CONNECTED),
    googleDriveAuthorizeURL: vi.fn().mockResolvedValue("https://accounts.example/auth"),
    googleDrivePickerToken: vi.fn().mockResolvedValue(TOKEN),
    googleDriveImportFile: vi.fn().mockResolvedValue({ uploaded: { file_id: "f" } }),
    ...overrides,
  };
  return client as unknown as ApiClient & typeof client;
};

const pick = (files: PickedDriveFile[] | null): PickerOpener => vi.fn().mockResolvedValue(files);

const renderFlow = (
  client: ReturnType<typeof fakeClient>,
  openPicker: PickerOpener,
  extra: Partial<{ onOpenChange: (open: boolean) => void; onImported: () => void }> = {}
) => {
  const onOpenChange = extra.onOpenChange ?? vi.fn();
  const onImported = extra.onImported ?? vi.fn();
  render(
    <GoogleDriveImportFlow
      open
      onOpenChange={onOpenChange}
      apiClient={client}
      onImported={onImported}
      openPicker={openPicker}
    />
  );
  return { onOpenChange, onImported };
};

describe("GoogleDriveImportFlow", () => {
  it("shows the unavailable state when the deployment has no Google credentials", async () => {
    const client = fakeClient({
      googleDriveStatus: vi.fn().mockResolvedValue({ enabled: false, connected: false }),
    });
    renderFlow(client, pick(null));
    await screen.findByText(/isn't configured on this deployment/);
    expect(client.googleDrivePickerToken).not.toHaveBeenCalled();
  });

  it("imports picked files two at a time, each with its own outcome", async () => {
    const gates = new Map<string, Deferred<unknown>>();
    const importFile = vi.fn((fileId: string) => {
      const gate = deferred<unknown>();
      gates.set(fileId, gate);
      return gate.promise;
    });
    const client = fakeClient({ googleDriveImportFile: importFile });
    const files: PickedDriveFile[] = [
      { id: "a", name: "atlas.tif", sizeBytes: 2048 },
      { id: "b", name: "brains.zarr.zip" },
      { id: "c", name: "counts.csv" },
    ];
    const { onOpenChange, onImported } = renderFlow(client, pick(files));

    // Concurrency is bounded: with three queued, only two start.
    await waitFor(() => expect(importFile).toHaveBeenCalledTimes(2));
    expect(importFile.mock.calls.map((call) => call[0])).toEqual(["a", "b"]);
    await screen.findByText("atlas.tif");
    expect(document.querySelectorAll('.gdrive-file[data-status="importing"]')).toHaveLength(2);
    expect(document.querySelectorAll('.gdrive-file[data-status="queued"]')).toHaveLength(1);

    // One slot frees, the third file starts; the finished one reads Imported.
    gates.get("a")!.resolve({ uploaded: { file_id: "fa" } });
    await waitFor(() => expect(importFile).toHaveBeenCalledTimes(3));
    await screen.findByText("Imported");

    gates.get("b")!.resolve({ uploaded: { file_id: "fb" } });
    gates.get("c")!.resolve({ uploaded: { file_id: "fc" } });
    await screen.findByText("3 files imported.");

    // Closing after success reports the import exactly once.
    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(onImported).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("retries only the failed files", async () => {
    const attempts: string[] = [];
    const importFile = vi.fn((fileId: string) => {
      attempts.push(fileId);
      if (fileId === "flaky" && attempts.filter((id) => id === "flaky").length === 1) {
        return Promise.reject(new Error("download was corrupted, retry"));
      }
      return Promise.resolve({ uploaded: { file_id: `f-${fileId}` } });
    });
    const client = fakeClient({ googleDriveImportFile: importFile });
    renderFlow(
      client,
      pick([
        { id: "solid", name: "solid.tif" },
        { id: "flaky", name: "flaky.tif" },
      ])
    );

    await screen.findByText("1 of 2 imported — the rest can be retried.");
    expect(screen.getByText("download was corrupted, retry")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Retry 1 failed/ }));
    await screen.findByText("2 files imported.");
    // The already-imported file was never re-sent.
    expect(attempts).toEqual(["solid", "flaky", "flaky"]);
  });

  it("returns to connect with guidance when Drive access needs a reconnect", async () => {
    const client = fakeClient({
      googleDrivePickerToken: vi.fn().mockRejectedValue(new Error("reconnect_required")),
    });
    renderFlow(client, pick(null));
    await screen.findByText("Google Drive access expired — reconnect your account.");
    expect(screen.getByRole("button", { name: "Connect Google Drive" })).toBeInTheDocument();
  });

  it("surfaces a slow picker failure as a retriable error, not a hang", async () => {
    // The rejection lands macrotasks after the phase has advanced to
    // "picking" — the timing where an effect-cleanup cancelled flag would
    // swallow the failure and hang the dialog on the spinner forever.
    let attempts = 0;
    const openPicker: PickerOpener = vi.fn(
      () =>
        new Promise<PickedDriveFile[] | null>((resolve, reject) => {
          attempts += 1;
          if (attempts === 1) {
            setTimeout(() => reject(new Error("Google's picker did not finish opening — retry.")), 20);
          } else {
            setTimeout(() => resolve([{ id: "late", name: "late.tif" }]), 5);
          }
        })
    );
    const client = fakeClient();
    renderFlow(client, openPicker);

    await screen.findByText("Google's picker did not finish opening — retry.");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    // The retry restarts the journey — new token, new picker — and a late
    // success still flows into the import list.
    await screen.findByText("1 file imported.");
    expect(client.googleDrivePickerToken).toHaveBeenCalledTimes(2);
  });

  it("closes quietly when the picker is cancelled", async () => {
    const client = fakeClient();
    const { onOpenChange, onImported } = renderFlow(client, pick(null));
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
    expect(onImported).not.toHaveBeenCalled();
  });

  it("offers a same-URL fallback link when the popup is blocked, and finishes on postMessage", async () => {
    const statuses = [
      { enabled: true, connected: false } as GoogleDriveStatus,
      CONNECTED,
    ];
    const client = fakeClient({
      googleDriveStatus: vi.fn(() => Promise.resolve(statuses.length > 1 ? statuses.shift()! : statuses[0])),
      googleDrivePickerToken: vi.fn(() => new Promise(() => {})), // hold at "Opening your Drive…"
    });
    const openSpy = vi.spyOn(window, "open").mockReturnValue(null); // the blocked-popup path
    try {
      renderFlow(client, pick(null));
      fireEvent.click(await screen.findByRole("button", { name: "Connect Google Drive" }));

      const fallback = await screen.findByRole("link", { name: /continue here/ });
      expect(fallback).toHaveAttribute("href", "https://accounts.example/auth");
      expect(openSpy).toHaveBeenCalledWith("about:blank", "ultra-google-drive-auth", expect.any(String));

      // Messages from foreign origins are ignored…
      fireEvent(
        window,
        new MessageEvent("message", {
          data: { type: "ultra-google-drive", status: "connected" },
          origin: "https://evil.example",
        })
      );
      expect(screen.getByText(/Finish connecting in the Google window/)).toBeInTheDocument();

      // …while the real callback's message advances the flow to the picker.
      fireEvent(
        window,
        new MessageEvent("message", {
          data: { type: "ultra-google-drive", status: "connected" },
          origin: window.location.origin,
        })
      );
      await screen.findByText("Opening your Drive…");
      expect(client.googleDrivePickerToken).toHaveBeenCalledTimes(1);
    } finally {
      openSpy.mockRestore();
    }
  });

  it("keeps the picker script lazy and the popup synchronous", () => {
    // The Google script must never load at module scope — only inside the
    // picker opener, after the user is connected.
    const beforeLoader = componentSource.slice(0, componentSource.indexOf("const loadPickerScript"));
    expect(beforeLoader).not.toContain("createElement(\"script\")");
    expect(componentSource.match(/apis\.google\.com/g)).toHaveLength(1);
    // Popup blockers only allow window.open on the click's own tick, so the
    // open must precede the authorize-URL await.
    const openIndex = componentSource.indexOf('window.open("about:blank"');
    const awaitIndex = componentSource.indexOf("await apiClient.googleDriveAuthorizeURL()");
    expect(openIndex).toBeGreaterThan(-1);
    expect(awaitIndex).toBeGreaterThan(openIndex);
    // A picker frame that never boots must not strand Google's backdrop over
    // the app: the boot watchdog disposes it and rejects with a retriable error.
    expect(componentSource).toContain("PICKER_BOOT_TIMEOUT_MS");
    expect(componentSource).toContain("pickerHandle?.dispose?.()");
  });
});
