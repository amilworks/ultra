// Shared formatting for the Training page. Sentence case, no "N/A" (rule 8):
// empty timestamps render as an em-dash or a phrase chosen by the caller.

export const formatTimestamp = (value: string | null | undefined): string => {
  const token = String(value ?? "").trim();
  if (!token) {
    return "—";
  }
  const date = new Date(token);
  if (Number.isNaN(date.getTime())) {
    return token;
  }
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
};

export const formatDay = (value: string | null | undefined): string => {
  const token = String(value ?? "").trim();
  if (!token) {
    return "—";
  }
  const date = new Date(token);
  if (Number.isNaN(date.getTime())) {
    return token;
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

export const timeAgo = (value: string | null | undefined, now: Date = new Date()): string => {
  const token = String(value ?? "").trim();
  if (!token) {
    return "—";
  }
  const then = new Date(token).getTime();
  if (!Number.isFinite(then)) {
    return token;
  }
  const minutes = Math.max(0, Math.round((now.getTime() - then) / 60000));
  if (minutes < 60) {
    return `${minutes} min ago`;
  }
  const hours = Math.round(minutes / 60);
  if (hours < 48) {
    return `${hours} h ago`;
  }
  return `${Math.round(hours / 24)} d ago`;
};

export const formatCount = (value: number): string =>
  new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(Number.isFinite(value) ? value : 0);

export const shortHash = (value: string | null | undefined): string => {
  const token = String(value ?? "").trim();
  return token ? `${token.slice(0, 6)}…` : "";
};

export const formatMetric = (value: number | null): string =>
  value == null ? "—" : String(Math.round(value * 1000) / 1000);
