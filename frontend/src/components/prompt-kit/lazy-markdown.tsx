import { lazy, type ComponentType } from "react";

import { retryDynamicImport } from "@/lib/lazy-retry";
import type { MarkdownProps } from "./markdown";

export const loadMarkdownRenderer = () => import("./markdown");

export const LazyMarkdown = lazy(() =>
  retryDynamicImport(loadMarkdownRenderer, { label: "Markdown" }).then((module) => ({
    default: module.Markdown as ComponentType<MarkdownProps>,
  }))
);
