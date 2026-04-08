import { useEffect, useState } from "react";

export function useLocalStorageState<T>(
  key: string,
  initialValue: T
): [T, (nextValue: T) => void] {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") {
      return initialValue;
    }
    try {
      const raw = window.localStorage.getItem(key);
      if (!raw) {
        return initialValue;
      }
      return JSON.parse(raw) as T;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Ignore local storage write failures.
    }
  }, [key, value]);

  return [value, setValue];
}

