# Google Drive import — operator setup

Ultra's "Add from Google Drive" flow uses the Google Picker: the user picks
files in Google's own UI, and the control plane copies each picked file into
Ultra's normal upload pipeline (dedup, catalog, previews — identical to a
browser upload). Ultra requests only the **`drive.file`** scope, which grants
access solely to files the user explicitly picks — never the rest of their
Drive. That scope is classified *sensitive* (not *restricted*), so no CASA
security audit is required for verification.

The feature is invisible until `ULTRA_CONTROL_GOOGLE_CLIENT_ID` is set, so
nothing below blocks a deploy.

## One-time Google Cloud setup (~10 minutes)

1. **Project** — console.cloud.google.com → create (or reuse) a project,
   e.g. `bisque-ultra`.
2. **Enable APIs** — *APIs & Services → Library*: enable **Google Drive API**
   and **Google Picker API**.
3. **OAuth consent screen** — *APIs & Services → OAuth consent screen*:
   - User type **External** (or **Internal** if the Workspace org covers all
     users — Internal skips verification entirely).
   - App name "BisQue Ultra", support + developer contact emails.
   - Scopes: add `https://www.googleapis.com/auth/drive.file` (plus the
     default `openid` / `email`).
   - While the app is in **Testing** mode, add each user as a test user
     (100 max, tokens expire after 7 days). Publish to **In production**
     when ready — `drive.file` being merely sensitive means Google's review
     is the lightweight one (no CASA audit).
4. **OAuth client** — *Credentials → Create credentials → OAuth client ID*:
   - Type **Web application**.
   - Authorized redirect URI (exact match, one per deployment):
     `https://ultra.ece.ucsb.edu/v2/integrations/google/callback`
   - Copy the client ID and client secret.
5. **Picker API key** — *Credentials → Create credentials → API key*:
   - Restrict it: *Application restrictions → Websites* → add the app origin
     (`https://ultra.ece.ucsb.edu`); *API restrictions* → Google Picker API.
6. **Control-plane env** (see `deploy/env/ultra-backend.env.example`):

   ```
   ULTRA_CONTROL_GOOGLE_CLIENT_ID=<from step 4>
   ULTRA_CONTROL_GOOGLE_CLIENT_SECRET=<from step 4>
   ULTRA_CONTROL_GOOGLE_REDIRECT_URL=https://ultra.ece.ucsb.edu/v2/integrations/google/callback
   ULTRA_CONTROL_GOOGLE_PICKER_API_KEY=<from step 5>
   ```

   `ULTRA_CONTROL_SECRET_ENCRYPTION_KEY` must already be set — refresh tokens
   are AES-256-GCM-encrypted at rest with it. Restart the control plane.

## How it behaves (reliability contract)

- **Connect** happens once per user in a popup; the callback page completes
  the flow by `postMessage` *and* the app re-checks status when the window
  regains focus, so a lost message cannot strand the user. A blocked popup
  degrades to a same-URL link.
- **Imports are per-file and atomic** — one request per picked file. A control
  restart mid-batch fails only the in-flight file; every failure is retriable
  individually from the dialog.
- **Streamed + verified** — files stream to disk (no full-file buffering) and
  the Drive `md5Checksum` is verified when Google provides one; mismatches
  are rejected, never silently cataloged.
- **429/5xx retries** with `Retry-After` honored, three attempts per step.
- **Revocation** (user revokes in Google account settings) surfaces as
  `reconnect_required`; the dialog routes straight back to Connect.
- **Native Google docs** (Docs/Sheets/Slides) are rejected with guidance to
  download as a regular file first — silent format conversion would surprise
  scientists.
- Refresh tokens live only in Postgres (encrypted); the browser and workers
  never see them. Disconnect deletes the row and best-effort revokes at
  Google.

## Limits

- Per-file cap: `ULTRA_CONTROL_GOOGLE_MAX_IMPORT_BYTES` (default 10 GiB),
  enforced before and during download.
- Shared-drive files work (`supportsAllDrives`), folders don't (pick files).
- Write-back ("save from Ultra to Drive") is not built yet, but `drive.file`
  already covers files the app creates — a future export needs no new consent.
