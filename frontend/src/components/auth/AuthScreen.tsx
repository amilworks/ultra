import { useEffect, useMemo, useState, type FormEvent } from "react";
import { ArrowUpRight, LockKeyhole, UserRound } from "lucide-react";

import { BisqueMarkIcon } from "@/components/icons/BisqueMarkIcon";
import { useTextStream } from "@/components/prompt-kit/response-stream";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type AuthMode = "login" | "guest";

type AuthScreenProps = {
  bisqueRoot: string;
  bisqueHomeUrl?: string;
  loading: boolean;
  allowGuest?: boolean;
  errorMessage?: string | null;
  onAuthenticate: (payload: { username: string; password: string }) => Promise<void>;
  onContinueGuest: (payload: {
    name: string;
    email: string;
    affiliation: string;
  }) => Promise<void>;
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
  bisqueRoot,
  bisqueHomeUrl,
  loading,
  allowGuest = true,
  errorMessage,
  onAuthenticate,
  onContinueGuest,
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
  const effectiveMode: AuthMode = !allowGuest && mode === "guest" ? "login" : mode;
  const activeHeroPhrase = prefersReducedMotion ? heroPhrases[0] : heroPhrases[heroPhraseIndex];
  const { displayedText: streamedHeroText, isComplete: heroTextComplete } = useTextStream({
    textStream: activeHeroPhrase,
    speed: prefersReducedMotion ? 100 : 18,
    characterChunkSize: prefersReducedMotion ? activeHeroPhrase.length : 1,
  });
  const visibleHeroText = prefersReducedMotion ? activeHeroPhrase : streamedHeroText || "\u00a0";

  useEffect(() => {
    if (!allowGuest && mode === "guest") {
      setMode("login");
      setLocalError(null);
    }
  }, [allowGuest, mode]);

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

  const submitLabel = effectiveMode === "login" ? "Sign in" : "Continue as guest";
  const mergedError = localError || errorMessage || null;

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
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
        await onContinueGuest({
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
            <span>BisQue Ultra</span>
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
            <CardTitle>{effectiveMode === "login" ? "Welcome back" : "Continue as guest"}</CardTitle>
            <CardDescription>
              {effectiveMode === "login"
                ? "Sign in with your BisQue username and password."
                : "Continue without BisQue credentials. Some BisQue operations may be limited."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {allowGuest ? (
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
                  Continue as guest
                </Button>
              </div>
            ) : null}

            <form className="auth-form" onSubmit={handleSubmit}>
              {effectiveMode === "login" ? (
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

              <Button type="submit" disabled={loading} className="w-full rounded-xl">
                {loading ? "Authenticating…" : submitLabel}
              </Button>
            </form>
          </CardContent>
        </Card>
      </section>
    </main>
  );
}
