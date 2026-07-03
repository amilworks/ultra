import { Button } from "@/components/ui/button";
import { Loader } from "@/components/prompt-kit";
import { BisqueMarkIcon } from "@/components/icons/BisqueMarkIcon";

type WorkOSRedirectScreenProps = {
  checking: boolean;
  loading: boolean;
  errorMessage?: string | null;
  statusMessage?: string | null;
  onRetry: () => Promise<void> | void;
};

export function WorkOSRedirectScreen({
  checking,
  loading,
  errorMessage,
  statusMessage,
  onRetry,
}: WorkOSRedirectScreenProps) {
  const title = checking
    ? "Checking your session"
    : statusMessage
      ? "Account not yet available"
      : errorMessage
        ? "Unable to open WorkOS"
        : "Opening WorkOS sign in";
  const message =
    statusMessage ||
    errorMessage ||
    "Taking you directly to the BisQue Ultra sign-in page.";
  const canRetry = !checking && !loading && Boolean(errorMessage || statusMessage);

  return (
    <main className="grid min-h-svh place-items-center bg-background p-6 text-foreground">
      <section
        className="grid w-full max-w-sm gap-4 rounded-2xl border bg-card p-6 text-center shadow-sm"
        aria-live="polite"
      >
        <div className="mx-auto flex size-11 items-center justify-center rounded-xl bg-muted text-muted-foreground">
          <BisqueMarkIcon className="size-5" />
        </div>
        <div className="grid gap-1">
          <h1 className="text-base font-semibold tracking-normal">{title}</h1>
          <p className="text-sm leading-6 text-muted-foreground">{message}</p>
        </div>
        {canRetry ? (
          <Button type="button" className="h-9 rounded-xl" onClick={() => void onRetry()}>
            Try again
          </Button>
        ) : null}
      </section>
    </main>
  );
}

export function AuthScreenLoadingFallback() {
  return (
    <main className="grid min-h-svh place-items-center bg-background p-6 text-foreground">
      <section className="grid w-full max-w-sm gap-3 rounded-2xl border bg-card p-6 text-center shadow-sm">
        <Loader className="mx-auto size-5 animate-spin text-muted-foreground" />
        <h1 className="text-xl font-semibold tracking-tight">Opening sign in</h1>
        <p className="text-sm text-muted-foreground">
          Preparing BisQue Ultra.
        </p>
      </section>
    </main>
  );
}
