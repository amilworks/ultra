import { useEffect, useMemo, useState, type FormEvent } from "react";
import { ArrowUpRight, LockKeyhole, UserRound } from "lucide-react";

import { BisqueMarkIcon } from "@/components/icons/BisqueMarkIcon";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type AuthMode = "login" | "guest";

type AuthScreenProps = {
  bisqueRoot: string;
  bisqueHomeUrl?: string;
  loading: boolean;
  oidcEnabled?: boolean;
  allowGuest?: boolean;
  errorMessage?: string | null;
  onAuthenticate: (payload: { username: string; password: string }) => Promise<void>;
  onStartOidcLogin?: () => void;
  onContinueGuest: (payload: {
    name: string;
    email: string;
    affiliation: string;
  }) => Promise<void>;
};

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
  oidcEnabled = false,
  allowGuest = true,
  errorMessage,
  onAuthenticate,
  onStartOidcLogin,
  onContinueGuest,
}: AuthScreenProps) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [guestName, setGuestName] = useState("");
  const [guestEmail, setGuestEmail] = useState("");
  const [guestAffiliation, setGuestAffiliation] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const bisqueHost = useMemo(() => hostFromUrl(bisqueRoot), [bisqueRoot]);
  const bisqueHomeHref = useMemo(() => {
    const explicit = String(bisqueHomeUrl ?? "").trim();
    if (explicit) {
      return explicit;
    }
    return `${bisqueRoot}/client_service/`;
  }, [bisqueHomeUrl, bisqueRoot]);
  const effectiveMode: AuthMode = !allowGuest && mode === "guest" ? "login" : mode;

  useEffect(() => {
    if (!allowGuest && mode === "guest") {
      setMode("login");
      setLocalError(null);
    }
  }, [allowGuest, mode]);

  const submitLabel =
    effectiveMode === "login"
      ? oidcEnabled
        ? "Continue with BisQue SSO"
        : "Sign in"
      : "Continue as guest";
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

    if (oidcEnabled) {
      setLocalError(null);
      onStartOidcLogin?.();
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
          <h1>Connect your BisQue account</h1>
          <p>
            {oidcEnabled
              ? "Sign in through BisQue SSO to use the same account across services."
              : "Use the same credentials you use on BisQue. After sign-in, uploads, browsing, and tool calls run against your account."}
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
                ? oidcEnabled
                  ? "Continue with your BisQue SSO account."
                  : "Sign in with your BisQue username and password."
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
                oidcEnabled ? (
                  <p className="text-muted-foreground text-sm">
                    You will be redirected to Keycloak and returned here after authentication.
                  </p>
                ) : (
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
                )
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
                    placeholder="UCSB VRL"
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
