import { useEffect, useMemo, useState, type FormEvent } from "react";
import { ArrowUpRight, LockKeyhole, UserRound } from "lucide-react";

import { BrandWordmark } from "@/components/BrandWordmark";
import { BisqueMarkIcon } from "@/components/icons/BisqueMarkIcon";
import { useTextStream } from "@/components/prompt-kit/response-stream";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type AuthMode = "login" | "guest";

const requestAccountLabel = "Request an Account";

type AuthScreenProps = {
  authProvider?: "local" | "workos";
  bisqueRoot: string;
  bisqueHomeUrl?: string;
  loading: boolean;
  allowGuest?: boolean;
  errorMessage?: string | null;
  statusMessage?: string | null;
  onAuthenticate: (payload: { username: string; password: string }) => Promise<void>;
  onStartHostedAuth?: () => Promise<void>;
  onRequestAccount: (payload: {
    name: string;
    email: string;
    affiliation: string;
  }) => Promise<void> | void;
};

const heroPhrases = [
  "Build the future",
  "What are we building today?",
  "What are we working on?",
  "Launch the next discovery",
] as const;

const HERO_PHRASE_DWELL_MS = 12_000;

const getPrefersReducedMotion = (): boolean =>
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const hostFromUrl = (value: string): string => {
  try {
    return new URL(value).host;
  } catch {
    return value;
  }
};

export function AuthScreen({
  authProvider = "local",
  bisqueRoot,
  bisqueHomeUrl,
  loading,
  allowGuest = true,
  errorMessage,
  statusMessage,
  onAuthenticate,
  onStartHostedAuth,
  onRequestAccount,
}: AuthScreenProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [guestName, setGuestName] = useState("");
  const [guestEmail, setGuestEmail] = useState("");
  const [guestAffiliation, setGuestAffiliation] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const [heroPhraseIndex, setHeroPhraseIndex] = useState(0);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(getPrefersReducedMotion);
  const bisqueHost = useMemo(() => hostFromUrl(bisqueRoot), [bisqueRoot]);
  const bisqueHomeHref = useMemo(() => {
    const explicit = String(bisqueHomeUrl ?? "").trim();
    if (explicit) {
      return explicit;
    }
    return `${bisqueRoot}/client_service/`;
  }, [bisqueHomeUrl, bisqueRoot]);
  const hostedAuth = authProvider === "workos";
  const effectiveMode: AuthMode = hostedAuth || (!allowGuest && mode === "guest") ? "login" : mode;
  const activeHeroPhrase = prefersReducedMotion ? heroPhrases[0] : heroPhrases[heroPhraseIndex];
  const { displayedText: streamedHeroText, isComplete: heroTextComplete } = useTextStream({
    textStream: activeHeroPhrase,
    speed: prefersReducedMotion ? 100 : 18,
    characterChunkSize: prefersReducedMotion ? activeHeroPhrase.length : 1,
  });
  const visibleHeroText = prefersReducedMotion ? activeHeroPhrase : streamedHeroText || "\u00a0";

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return undefined;
    }
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handleChange = () => {
      setPrefersReducedMotion(media.matches);
    };
    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  useEffect(() => {
    if (prefersReducedMotion || !heroTextComplete) {
      return undefined;
    }
    const timeoutId = window.setTimeout(() => {
      setHeroPhraseIndex((currentIndex) => (currentIndex + 1) % heroPhrases.length);
    }, HERO_PHRASE_DWELL_MS);
    return () => window.clearTimeout(timeoutId);
  }, [heroTextComplete, prefersReducedMotion]);

  const submitLabel = hostedAuth
    ? "Sign in with WorkOS"
    : effectiveMode === "login"
      ? "Sign in"
      : requestAccountLabel;
  const visibleLocalError = effectiveMode === mode ? localError : null;
  const mergedError = visibleLocalError || errorMessage || null;

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (hostedAuth) {
      setLocalError(null);
      try {
        await onStartHostedAuth?.();
      } catch {
        // Parent component exposes API error state.
      }
      return;
    }
    if (effectiveMode === "guest") {
      const nextName = guestName.trim();
      const nextEmail = guestEmail.trim();
      const nextAffiliation = guestAffiliation.trim();
      if (!nextName || !nextEmail || !nextAffiliation) {
        setLocalError("Name, email, and affiliation are required.");
        return;
      }
      setLocalError(null);
      try {
        await onRequestAccount({
          name: nextName,
          email: nextEmail,
          affiliation: nextAffiliation,
        });
      } catch {
        // Parent component exposes API error state.
      }
      return;
    }

    const nextUsername = username.trim();
    const nextPassword = password.trim();
    if (!nextUsername || !nextPassword) {
      setLocalError("Username and password are required.");
      return;
    }
    setLocalError(null);
    try {
      await onAuthenticate({ username: nextUsername, password: nextPassword });
    } catch {
      // Parent component exposes API error state.
    }
  };

  return (
    <main className="auth-screen">
      <section className="auth-screen-hero">
        <div className="auth-screen-hero-overlay">
          <div className="auth-screen-logo">
            <div className="auth-screen-logo-mark">
              <BisqueMarkIcon className="size-5" />
            </div>
            <BrandWordmark />
          </div>
          <h1 className="auth-hero-title" aria-label={heroPhrases[0]}>
            <span className="auth-hero-typewriter" aria-hidden="true">
              <span className="auth-hero-typewriter-text">{visibleHeroText}</span>
              <span className="auth-hero-typewriter-caret" />
            </span>
          </h1>
          <p>
            Sign in to bring your BisQue data, uploads, browsing, and tool calls into one
            workspace.
          </p>
          <a href={bisqueHomeHref} target="_blank" rel="noreferrer">
            Open {bisqueHost}
            <ArrowUpRight className="size-4" />
          </a>
        </div>
      </section>

      <section className="auth-screen-form">
        <Card className="auth-card">
          <CardHeader>
            <CardTitle>
              {hostedAuth
                ? "Welcome back"
                : effectiveMode === "login"
                  ? "Welcome back"
                  : requestAccountLabel}
            </CardTitle>
            <CardDescription>
              {hostedAuth
                ? "Sign in with your organization account."
                : effectiveMode === "login"
                  ? "Sign in with your BisQue username and password."
                  : "Share your name, email, and affiliation so an administrator can review access."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {allowGuest && !hostedAuth ? (
              <div className="auth-mode-toggle" role="tablist" aria-label="Authentication mode">
                <Button
                  type="button"
                  variant={mode === "login" ? "default" : "ghost"}
                  className={cn("h-9 rounded-lg", mode === "login" ? "" : "text-muted-foreground")}
                  onClick={() => {
                    setMode("login");
                    setLocalError(null);
                  }}
                >
                  Sign in
                </Button>
                <Button
                  type="button"
                  variant={mode === "guest" ? "default" : "ghost"}
                  className={cn("h-9 rounded-lg", mode === "guest" ? "" : "text-muted-foreground")}
                  onClick={() => {
                    setMode("guest");
                    setLocalError(null);
                  }}
                >
                  {requestAccountLabel}
                </Button>
              </div>
            ) : null}

            <form className="auth-form" onSubmit={handleSubmit}>
              {hostedAuth ? (
                <Button
                  type="submit"
                  className="h-9 w-full rounded-xl"
                  disabled={loading}
                >
                  <LockKeyhole className="size-4" />
                  {loading ? "Opening sign in..." : submitLabel}
                </Button>
              ) : effectiveMode === "login" ? (
                <>
                  <label className="auth-label" htmlFor="bisque-username">
                    <UserRound className="size-4" />
                    Username
                  </label>
                  <Input
                    id="bisque-username"
                    autoComplete="username"
                    placeholder="your.username"
                    value={username}
                    onChange={(event) => setUsername(event.target.value)}
                    disabled={loading}
                  />

                  <label className="auth-label" htmlFor="bisque-password">
                    <LockKeyhole className="size-4" />
                    Password
                  </label>
                  <Input
                    id="bisque-password"
                    type="password"
                    autoComplete="current-password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    disabled={loading}
                  />
                </>
              ) : (
                <>
                  <label className="auth-label" htmlFor="guest-name">
                    <UserRound className="size-4" />
                    Name
                  </label>
                  <Input
                    id="guest-name"
                    autoComplete="name"
                    placeholder="Your full name"
                    value={guestName}
                    onChange={(event) => setGuestName(event.target.value)}
                    disabled={loading}
                  />

                  <label className="auth-label" htmlFor="guest-email">
                    <UserRound className="size-4" />
                    Email
                  </label>
                  <Input
                    id="guest-email"
                    type="email"
                    autoComplete="email"
                    placeholder="you@institution.edu"
                    value={guestEmail}
                    onChange={(event) => setGuestEmail(event.target.value)}
                    disabled={loading}
                  />

                  <label className="auth-label" htmlFor="guest-affiliation">
                    <UserRound className="size-4" />
                    Affiliation
                  </label>
                  <Input
                    id="guest-affiliation"
                    autoComplete="organization"
                    placeholder="Research lab, company, or university"
                    value={guestAffiliation}
                    onChange={(event) => setGuestAffiliation(event.target.value)}
                    disabled={loading}
                  />
                </>
              )}

              {mergedError ? <p className="auth-error">{mergedError}</p> : null}
              {!mergedError && statusMessage ? (
                <p className="auth-status-message">{statusMessage}</p>
              ) : null}

              {!hostedAuth ? (
                <Button type="submit" disabled={loading} className="w-full rounded-xl">
                  {loading
                    ? effectiveMode === "guest"
                      ? "Submitting request…"
                      : "Authenticating…"
                    : submitLabel}
                </Button>
              ) : null}
            </form>
          </CardContent>
        </Card>
      </section>
    </main>
  );
}
