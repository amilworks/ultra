import { useEffect, useState } from "react";

export type ThemePreference = "system" | "light" | "dark";

// Applies the user's theme preference to the document (class, color-scheme,
// data-theme on both <html> and <body>), tracks the OS scheme while in
// "system", and returns the resolved concrete theme.
export function useThemePreference(themePreference: ThemePreference): "light" | "dark" {
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = (): void => {
      const shouldUseDark =
        themePreference === "dark" ||
        (themePreference === "system" && mediaQuery.matches);
      document.documentElement.classList.toggle("dark", shouldUseDark);
      document.body.classList.toggle("dark", shouldUseDark);
      document.documentElement.style.colorScheme = shouldUseDark ? "dark" : "light";
      document.body.style.colorScheme = shouldUseDark ? "dark" : "light";
      document.documentElement.setAttribute(
        "data-theme",
        shouldUseDark ? "dark" : "light"
      );
      document.body.setAttribute("data-theme", shouldUseDark ? "dark" : "light");
      setResolvedTheme(shouldUseDark ? "dark" : "light");
    };
    applyTheme();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", applyTheme);
    } else {
      mediaQuery.addListener(applyTheme);
    }
    return () => {
      if (typeof mediaQuery.removeEventListener === "function") {
        mediaQuery.removeEventListener("change", applyTheme);
      } else {
        mediaQuery.removeListener(applyTheme);
      }
    };
  }, [themePreference]);

  return resolvedTheme;
}
