export const formatBytes = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "0 B";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = -1;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[index]}`;
};

export const formatTokens = (tokens: number): string => {
  if (!Number.isFinite(tokens) || tokens <= 0) {
    return "0";
  }
  if (tokens < 1000) {
    return String(Math.round(tokens));
  }
  const units = ["K", "M", "B", "T"];
  let value = tokens;
  let index = -1;
  while (value >= 1000 && index < units.length - 1) {
    value /= 1000;
    index += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)}${units[index]}`;
};

export const formatDurationSeconds = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "—";
  }
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
};

export const fileExt = (name: string): string => {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot + 1).toLowerCase() : "";
};

export const shortId = (value: string): string => {
  if (!value) return "";
  if (value.length <= 12) return value;
  return `${value.slice(0, 6)}...${value.slice(-4)}`;
};

export const toLocalDateTime = (iso?: string): string => {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
};

