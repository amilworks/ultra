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
      // Browser-chrome tint. The media-scoped metas in index.html cover the
      // OS-driven case before hydration; an explicit in-app choice has to
      // override BOTH of them, media attribute and all, or an OS-light device
      // set to Dark keeps a paper-white address bar. Grounds, not hexes from
      // anywhere else: these are --bg-main in each theme block.
      const ground = shouldUseDark ? "#0b0e11" : "#f2f3f3";
      document
        .querySelectorAll('meta[name="theme-color"]')
        .forEach((meta) => meta.setAttribute("content", ground));
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
