import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Check,
  CloudDownload,
  ExternalLink,
  Loader2,
  RotateCcw,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ApiClient } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

/* eslint react-hooks/set-state-in-effect: "off" -- This dialog is a flow
   state machine: its effects ARE the drivers (status check on open, picker
   journey, bounded import queue), and `phase`/`files` advancing inside them is
   the mechanism, not derived state. The rule's positional reporting would
   otherwise demand a disable on every traced call site. */

// Add-from-Google-Drive, picker model. One dialog owns the whole journey —
// connect (popup OAuth), pick (Google's Picker), import (per-file, each an
// atomic server-side pull) — so the user never wonders what happens next.
//
// Reliability posture mirrors the backend's:
// - The popup is opened SYNCHRONOUSLY on the click (popup blockers punish
//   async opens); if it is blocked anyway, a same-gesture link takes over.
// - Connect completion arrives by postMessage from the callback page AND by
//   a status re-check on window focus — either signal completes the step.
// - Imports run two at a time; every file holds its own state, one failure
//   never touches its neighbors, and Retry re-sends exactly the failed ones.
// - The picker script loads lazily with a timeout and a retriable error
//   state — a CDN hiccup is a message, never a hung dialog.

const PICKER_SCRIPT_URL = "https://apis.google.com/js/api.js";
const PICKER_LOAD_TIMEOUT_MS = 12_000;
const PICKER_BOOT_TIMEOUT_MS = 15_000;
const CONNECT_TIMEOUT_MS = 180_000;
const IMPORT_CONCURRENCY = 2;

export type PickedDriveFile = {
  id: string;
  name: string;
  sizeBytes?: number;
  mimeType?: string;
};

type ImportFileState = PickedDriveFile & {
  status: "queued" | "importing" | "imported" | "failed";
  error?: string;
};

type FlowPhase =
  | "checking"
  | "unavailable"
  | "connect"
  | "authorizing"
  | "loading-picker"
  | "picking"
  | "picker-error"
  | "importing"
  | "done";

export type PickerOpener = (
  token: string,
  apiKey: string | undefined,
  appId: string | undefined
) => Promise<PickedDriveFile[] | null>;

declare global {
  interface Window {
    gapi?: {
      load: (name: string, callback: { callback: () => void; onerror?: () => void }) => void;
    };
    google?: {
      picker?: GooglePickerNamespace;
    };
  }
}

type GooglePickerNamespace = {
  PickerBuilder: new () => GooglePickerBuilder;
  DocsView: new () => GooglePickerDocsView;
  Feature: { MULTISELECT_ENABLED: unknown };
  Action: { PICKED: string; CANCEL: string };
  Response: { ACTION: string; DOCUMENTS: string };
  Document: { ID: string; NAME: string; SIZE_BYTES: string; MIME_TYPE: string };
};

type GooglePickerBuilder = {
  addView(view: GooglePickerDocsView): GooglePickerBuilder;
  setOAuthToken(token: string): GooglePickerBuilder;
  setDeveloperKey(key: string): GooglePickerBuilder;
  setAppId(appId: string): GooglePickerBuilder;
  setOrigin(origin: string): GooglePickerBuilder;
  enableFeature(feature: unknown): GooglePickerBuilder;
  setCallback(callback: (data: Record<string, unknown>) => void): GooglePickerBuilder;
  build(): { setVisible(visible: boolean): void; dispose?: () => void };
};

type GooglePickerDocsView = {
  setIncludeFolders(include: boolean): GooglePickerDocsView;
  setSelectFolderEnabled(enabled: boolean): GooglePickerDocsView;
};

let pickerScriptPromise: Promise<void> | null = null;

const loadPickerScript = (): Promise<void> => {
  if (window.google?.picker) {
    return Promise.resolve();
  }
  if (!pickerScriptPromise) {
    pickerScriptPromise = new Promise<void>((resolve, reject) => {
      const fail = (message: string) => {
        pickerScriptPromise = null; // a later attempt gets a fresh try
        reject(new Error(message));
      };
      const timeout = window.setTimeout(
        () => fail("Google's picker script took too long to load — check your connection and retry."),
        PICKER_LOAD_TIMEOUT_MS
      );
      const onReady = () => {
        window.gapi?.load("picker", {
          callback: () => {
            window.clearTimeout(timeout);
            resolve();
          },
          onerror: () => {
            window.clearTimeout(timeout);
            fail("Google's picker module failed to initialize — retry.");
          },
        });
      };
      const existing = document.querySelector<HTMLScriptElement>(
        `script[src="${PICKER_SCRIPT_URL}"]`
      );
      if (existing) {
        if (window.gapi) {
          onReady();
          return;
        }
        existing.addEventListener("load", onReady, { once: true });
        return;
      }
      const script = document.createElement("script");
      script.src = PICKER_SCRIPT_URL;
      script.async = true;
      script.onload = onReady;
      script.onerror = () => {
        window.clearTimeout(timeout);
        fail("Google's picker script could not be loaded — check your connection and retry.");
      };
      document.head.appendChild(script);
    });
  }
  return pickerScriptPromise;
};

const openGooglePicker: PickerOpener = async (token, apiKey, appId) => {
  await loadPickerScript();
  const picker = window.google?.picker;
  if (!picker) {
    throw new Error("Google's picker is unavailable — retry.");
  }
  return new Promise((resolve, reject) => {
    // If Google's frame never boots (bad token, blocked iframe, network drop),
    // no callback ever fires and Google's backdrop would cover the app with no
    // way out. Any callback at all — LOADED included — proves life; without
    // one we dispose Google's DOM ourselves and turn it into a retriable error.
    let pickerHandle: { setVisible(visible: boolean): void; dispose?: () => void } | null = null;
    let sawLife = false;
    const bootWatchdog = window.setTimeout(() => {
      if (!sawLife) {
        try {
          pickerHandle?.dispose?.();
        } catch {
          // Disposal is best-effort; the error below is what the user needs.
        }
        reject(new Error("Google's picker did not finish opening — retry."));
      }
    }, PICKER_BOOT_TIMEOUT_MS);
    const view = new picker.DocsView().setIncludeFolders(false).setSelectFolderEnabled(false);
    let builder = new picker.PickerBuilder()
      .addView(view)
      .setOAuthToken(token)
      .setOrigin(window.location.origin)
      .enableFeature(picker.Feature.MULTISELECT_ENABLED)
      .setCallback((data) => {
        sawLife = true;
        const action = data[picker.Response.ACTION];
        if (action === picker.Action.PICKED) {
          window.clearTimeout(bootWatchdog);
          const documents = (data[picker.Response.DOCUMENTS] as Array<Record<string, unknown>>) ?? [];
          resolve(
            documents.map((entry) => ({
              id: String(entry[picker.Document.ID] ?? ""),
              name: String(entry[picker.Document.NAME] ?? "Drive file"),
              sizeBytes: Number(entry[picker.Document.SIZE_BYTES]) || undefined,
              mimeType: typeof entry[picker.Document.MIME_TYPE] === "string"
                ? (entry[picker.Document.MIME_TYPE] as string)
                : undefined,
            }))
          );
        } else if (action === picker.Action.CANCEL) {
          window.clearTimeout(bootWatchdog);
          resolve(null);
        }
      });
    if (apiKey) {
      builder = builder.setDeveloperKey(apiKey);
    }
    if (appId) {
      builder = builder.setAppId(appId);
    }
    pickerHandle = builder.build();
    pickerHandle.setVisible(true);
  });
};

export type GoogleDriveImportFlowProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  apiClient: ApiClient;
  /** Called after at least one file imported, so the resource list refreshes. */
  onImported: () => void;
  /** Test seam; defaults to the real Google Picker. */
  openPicker?: PickerOpener;
};

export function GoogleDriveImportFlow({
  open,
  onOpenChange,
  apiClient,
  onImported,
  openPicker = openGooglePicker,
}: GoogleDriveImportFlowProps) {
  const [phase, setPhase] = useState<FlowPhase>("checking");
  const [accountEmail, setAccountEmail] = useState<string>("");
  const [flowError, setFlowError] = useState<string | null>(null);
  const [blockedAuthorizeURL, setBlockedAuthorizeURL] = useState<string | null>(null);
  const [files, setFiles] = useState<ImportFileState[]>([]);
  const importedAnyRef = useRef(false);
  const generationRef = useRef(0);
  const connectCleanupRef = useRef<(() => void) | null>(null);

  const closeFlow = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen) {
        generationRef.current += 1; // invalidate in-flight async work
        connectCleanupRef.current?.();
        if (importedAnyRef.current) {
          onImported();
        }
        // Every close funnels through here (X, overlay, Escape, Done), so the
        // next open starts from a clean journey.
        setFiles([]);
        setFlowError(null);
        setBlockedAuthorizeURL(null);
        importedAnyRef.current = false;
      }
      onOpenChange(nextOpen);
    },
    [onImported, onOpenChange]
  );

  const checkStatus = useCallback(async () => {
    const generation = generationRef.current;
    setPhase("checking");
    setFlowError(null);
    try {
      const status = await apiClient.googleDriveStatus();
      if (generation !== generationRef.current) {
        return;
      }
      if (!status.enabled) {
        setPhase("unavailable");
        return;
      }
      setAccountEmail(status.account_email ?? "");
      if (!status.connected || status.status === "broken") {
        setPhase("connect");
        return;
      }
      setPhase("loading-picker");
    } catch (error) {
      if (generation !== generationRef.current) {
        return;
      }
      setFlowError(error instanceof Error ? error.message : String(error));
      setPhase("connect");
    }
  }, [apiClient]);

  // Opening the dialog starts the journey; closeFlow resets it on the way out.
  useEffect(() => {
    if (!open) {
      return;
    }
    generationRef.current += 1;
    void checkStatus();
  }, [open, checkStatus]);

  const armConnectListeners = useCallback(
    (generation: number) => {
      connectCleanupRef.current?.();
      let settled = false;
      const settle = (connected: boolean, message?: string) => {
        if (settled || generation !== generationRef.current) {
          return;
        }
        settled = true;
        cleanup();
        if (connected) {
          void checkStatus();
        } else {
          setFlowError(message ?? "Google Drive was not connected.");
          setPhase("connect");
        }
      };
      const onMessage = (event: MessageEvent) => {
        if (event.origin !== window.location.origin) {
          return;
        }
        const payload = event.data as { type?: string; status?: string };
        if (payload?.type !== "ultra-google-drive") {
          return;
        }
        settle(payload.status === "connected",
          payload.status === "denied" ? "Google Drive access was declined." : undefined);
      };
      // Belt for the popup path: if the message is lost (window closed early,
      // strict browser), a focus-triggered status check still completes.
      const onFocus = () => {
        void apiClient.googleDriveStatus().then((status) => {
          if (status.connected && status.status !== "broken") {
            settle(true);
          }
        }).catch(() => {});
      };
      const timeout = window.setTimeout(
        () => settle(false, "The connect window timed out — try again."),
        CONNECT_TIMEOUT_MS
      );
      const cleanup = () => {
        window.removeEventListener("message", onMessage);
        window.removeEventListener("focus", onFocus);
        window.clearTimeout(timeout);
        connectCleanupRef.current = null;
      };
      window.addEventListener("message", onMessage);
      window.addEventListener("focus", onFocus);
      connectCleanupRef.current = cleanup;
    },
    [apiClient, checkStatus]
  );

  const beginConnect = useCallback(async () => {
    const generation = generationRef.current;
    setFlowError(null);
    setBlockedAuthorizeURL(null);
    // Synchronous open under the user's click — popup blockers allow this;
    // the URL streams in afterwards.
    const popup = window.open("about:blank", "ultra-google-drive-auth", "width=520,height=640");
    let authorizeURL: string;
    try {
      authorizeURL = await apiClient.googleDriveAuthorizeURL();
    } catch (error) {
      popup?.close();
      if (generation === generationRef.current) {
        setFlowError(error instanceof Error ? error.message : String(error));
      }
      return;
    }
    if (generation !== generationRef.current) {
      popup?.close();
      return;
    }
    if (!popup || popup.closed) {
      // Popup blocked: hand the same URL to a real link the user can click —
      // that click is its own gesture, which blockers respect.
      setBlockedAuthorizeURL(authorizeURL);
      setPhase("authorizing");
      armConnectListeners(generation);
      return;
    }
    popup.location.replace(authorizeURL);
    setPhase("authorizing");
    armConnectListeners(generation);
  }, [apiClient, armConnectListeners]);

  // loading-picker: mint a token, open Google's picker, collect the choice.
  // Staleness is judged by generation (bumped when the dialog closes), NOT by
  // an effect-cleanup flag: this effect advances `phase` itself mid-journey,
  // which re-runs the effect, and a cleanup flag would mark the still-live
  // picker promise as cancelled — its files or failure would vanish silently.
  useEffect(() => {
    if (!open || phase !== "loading-picker") {
      return;
    }
    const generation = generationRef.current;
    void (async () => {
      try {
        const token = await apiClient.googleDrivePickerToken();
        if (generation !== generationRef.current) {
          return;
        }
        setPhase("picking");
        const picked = await openPicker(
          token.access_token,
          token.picker_api_key || undefined,
          token.app_id || undefined
        );
        if (generation !== generationRef.current) {
          return;
        }
        if (!picked || picked.length === 0) {
          closeFlow(false);
          return;
        }
        setFiles(picked.map((file) => ({ ...file, status: "queued" as const })));
        setPhase("importing");
      } catch (error) {
        if (generation !== generationRef.current) {
          return;
        }
        const message = error instanceof Error ? error.message : String(error);
        if (message.includes("reconnect")) {
          setPhase("connect");
          setFlowError("Google Drive access expired — reconnect your account.");
          return;
        }
        // Token minting or the picker itself hiccupped; the account is still
        // connected, so offer a plain retry rather than a re-connect.
        setFlowError(message);
        setPhase("picker-error");
      }
    })();
  }, [open, phase, apiClient, openPicker, closeFlow]);

  // importing: two at a time, each file an independent atomic outcome.
  // This effect IS the queue driver: `files` is the only source of truth, and
  // every settled import re-renders and re-evaluates it. The setState calls
  // below are the queue advancing, not derived state — hence the targeted
  // lint exemptions.
  useEffect(() => {
    if (phase !== "importing") {
      return;
    }
    const importing = files.filter((file) => file.status === "importing").length;
    const nextQueued = files.find((file) => file.status === "queued");
    if (!nextQueued) {
      if (importing === 0) {
        setPhase("done");
      }
      return;
    }
    if (importing >= IMPORT_CONCURRENCY) {
      return;
    }
    const generation = generationRef.current;
    const target = nextQueued.id;
    setFiles((current) =>
      current.map((file) => (file.id === target ? { ...file, status: "importing" } : file))
    );
    void apiClient
      .googleDriveImportFile(target)
      .then(() => {
        importedAnyRef.current = true;
        if (generation !== generationRef.current) {
          return;
        }
        setFiles((current) =>
          current.map((file) => (file.id === target ? { ...file, status: "imported", error: undefined } : file))
        );
      })
      .catch((error: unknown) => {
        if (generation !== generationRef.current) {
          return;
        }
        setFiles((current) =>
          current.map((file) =>
            file.id === target
              ? {
                  ...file,
                  status: "failed",
                  error: error instanceof Error ? error.message : String(error),
                }
              : file
          )
        );
      });
  }, [phase, files, apiClient]);

  const retryFailed = useCallback(() => {
    setFiles((current) =>
      current.map((file) =>
        file.status === "failed" ? { ...file, status: "queued", error: undefined } : file
      )
    );
    setPhase("importing");
  }, []);

  const counts = useMemo(() => {
    const imported = files.filter((file) => file.status === "imported").length;
    const failed = files.filter((file) => file.status === "failed").length;
    return { imported, failed, total: files.length };
  }, [files]);

  return (
    <Dialog open={open} onOpenChange={closeFlow}>
      <DialogContent className="gdrive-dialog" data-testid="gdrive-flow">
        <DialogHeader>
          <DialogTitle>Add from Google Drive</DialogTitle>
          <DialogDescription>
            {phase === "done"
              ? counts.failed > 0
                ? `${counts.imported} of ${counts.total} imported — the rest can be retried.`
                : `${counts.imported} file${counts.imported === 1 ? "" : "s"} imported.`
              : phase === "importing"
                ? "Importing your selection. Each file lands as a normal Ultra resource."
                : `Pick files in your Drive; Ultra copies them in and they behave like any upload.${
                    accountEmail ? ` Connected as ${accountEmail}.` : ""
                  }`}
          </DialogDescription>
        </DialogHeader>

        {phase === "checking" ? (
          <div className="gdrive-state" role="status">
            <Loader2 className="animate-spin" data-icon="inline-start" />
            Checking your Google Drive connection…
          </div>
        ) : null}

        {phase === "unavailable" ? (
          <div className="gdrive-state">
            <AlertCircle data-icon="inline-start" />
            Google Drive isn&apos;t configured on this deployment. An administrator needs to set
            the Google OAuth credentials on the control plane.
          </div>
        ) : null}

        {phase === "connect" ? (
          <div className="gdrive-connect">
            {flowError ? <p className="gdrive-error" role="alert">{flowError}</p> : null}
            <p>
              Connect your Google account once. Ultra only ever sees the files you pick —
              nothing else in your Drive.
            </p>
            <Button type="button" onClick={() => void beginConnect()}>
              Connect Google Drive
            </Button>
          </div>
        ) : null}

        {phase === "authorizing" ? (
          <div className="gdrive-state" role="status">
            <Loader2 className="animate-spin" data-icon="inline-start" />
            Finish connecting in the Google window…
            {blockedAuthorizeURL ? (
              <a
                className="gdrive-popup-fallback"
                href={blockedAuthorizeURL}
                target="_blank"
                rel="noreferrer noopener"
              >
                Your browser blocked the window — continue here
                <ExternalLink data-icon="inline-end" />
              </a>
            ) : null}
          </div>
        ) : null}

        {phase === "loading-picker" || phase === "picking" ? (
          <div className="gdrive-state" role="status">
            <Loader2 className="animate-spin" data-icon="inline-start" />
            {phase === "loading-picker" ? "Opening your Drive…" : "Pick files in the Google window…"}
          </div>
        ) : null}

        {phase === "picker-error" ? (
          <div className="gdrive-connect">
            <p className="gdrive-error" role="alert">{flowError ?? "Google's picker could not open."}</p>
            <Button type="button" onClick={() => setPhase("loading-picker")}>
              <RotateCcw data-icon="inline-start" />
              Try again
            </Button>
          </div>
        ) : null}

        {(phase === "importing" || phase === "done") && files.length > 0 ? (
          <>
            <ul className="gdrive-file-list">
              {files.map((file) => (
                <li key={file.id} className="gdrive-file" data-status={file.status}>
                  <span className="gdrive-file-icon" aria-hidden="true">
                    {file.status === "imported" ? (
                      <Check />
                    ) : file.status === "failed" ? (
                      <AlertCircle />
                    ) : file.status === "importing" ? (
                      <Loader2 className="animate-spin" />
                    ) : (
                      <CloudDownload />
                    )}
                  </span>
                  <span className="gdrive-file-copy">
                    <span className="gdrive-file-name" title={file.name}>{file.name}</span>
                    <span className="gdrive-file-meta">
                      {file.status === "failed"
                        ? file.error ?? "Import failed"
                        : file.status === "imported"
                          ? "Imported"
                          : file.status === "importing"
                            ? `Importing${file.sizeBytes ? ` · ${formatBytes(file.sizeBytes)}` : ""}…`
                            : `Queued${file.sizeBytes ? ` · ${formatBytes(file.sizeBytes)}` : ""}`}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <div className="gdrive-actions">
              {counts.failed > 0 && phase === "done" ? (
                <Button type="button" variant="outline" onClick={retryFailed}>
                  <RotateCcw data-icon="inline-start" />
                  Retry {counts.failed} failed
                </Button>
              ) : null}
              <Button
                type="button"
                variant={phase === "done" ? "default" : "outline"}
                className={cn(phase !== "done" && "gdrive-close-secondary")}
                onClick={() => closeFlow(false)}
              >
                {/* Truthful copy: files already in flight finish on the server
                    either way; queued ones will not start after closing. */}
                {phase === "done" ? "Done" : "Stop after current files"}
              </Button>
            </div>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
