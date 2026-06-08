import { lazy, type ComponentType } from "react";

import type { MarkdownProps } from "./markdown";

export const loadMarkdownRenderer = () => import("./markdown");

export const LazyMarkdown = lazy(async () => {
  const module = await loadMarkdownRenderer();
  return {
    default: module.Markdown as ComponentType<MarkdownProps>,
  };
});
