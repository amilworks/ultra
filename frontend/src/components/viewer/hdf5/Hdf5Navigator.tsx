import { useEffect, useMemo, useRef, useState } from "react";
import { X } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { cn } from "@/lib/utils";
import type { Hdf5ViewerTreeNode } from "@/types";

import { formatSummaryToken } from "./formatters";

type Hdf5NavigatorProps = {
  tree: Hdf5ViewerTreeNode[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
  truncated: boolean;
};

type FlatDatasetEntry = {
  contextPath: string;
  name: string;
  path: string;
  searchText: string;
  sectionLabel: string;
  summaryText: string;
};

const countDatasets = (nodes: Hdf5ViewerTreeNode[]): number =>
  nodes.reduce((total, node) => {
    if (node.node_type === "dataset") {
      return total + 1;
    }
    return total + countDatasets(node.children ?? []);
  }, 0);

const buildNodeSubtitleParts = (node: Hdf5ViewerTreeNode): string[] =>
  [
    node.node_type === "dataset" && node.shape?.length ? node.shape.join(" x ") : null,
    node.dtype ?? null,
    node.preview_kind ? formatSummaryToken(node.preview_kind) : null,
  ].filter(Boolean) as string[];

const buildDatasetSectionLabel = (segments: string[]): string => {
  if (segments.length === 0) {
    return "Root";
  }
  if (segments[0] === "DataContainers" && segments[1]) {
    return segments[1];
  }
  return segments[0];
};

const buildSearchRank = (entry: FlatDatasetEntry, normalizedQuery: string): number => {
  const name = entry.name.toLowerCase();
  const contextPath = entry.contextPath.toLowerCase();
  const path = entry.path.toLowerCase();
  if (name === normalizedQuery) {
    return 0;
  }
  if (name.startsWith(normalizedQuery)) {
    return 1;
  }
  if (name.includes(normalizedQuery)) {
    return 2;
  }
  if (contextPath.includes(normalizedQuery)) {
    return 3;
  }
  if (path.includes(normalizedQuery)) {
    return 4;
  }
  return 5;
};

const filterDatasetEntries = (
  datasetEntries: FlatDatasetEntry[],
  query: string,
  selectedPath: string | null
): FlatDatasetEntry[] => {
  const trimmedQuery = query.trim();
  if (!trimmedQuery) {
    return datasetEntries;
  }
  const normalizedQuery = trimmedQuery.toLowerCase();
  return datasetEntries
    .filter((entry) => entry.searchText.includes(normalizedQuery))
    .sort((left, right) => {
      const rankDiff = buildSearchRank(left, normalizedQuery) - buildSearchRank(right, normalizedQuery);
      if (rankDiff !== 0) {
        return rankDiff;
      }
      if (left.path === selectedPath && right.path !== selectedPath) {
        return -1;
      }
      if (right.path === selectedPath && left.path !== selectedPath) {
        return 1;
      }
      return left.path.localeCompare(right.path);
    });
};

const resolveActivePath = (
  entries: FlatDatasetEntry[],
  currentPath: string | null,
  selectedPath: string | null
): string | null => {
  if (currentPath && entries.some((entry) => entry.path === currentPath)) {
    return currentPath;
  }
  if (selectedPath && entries.some((entry) => entry.path === selectedPath)) {
    return selectedPath;
  }
  return entries[0]?.path ?? null;
};

const flattenDatasets = (
  nodes: Hdf5ViewerTreeNode[],
  ancestors: string[] = []
): FlatDatasetEntry[] =>
  nodes.flatMap((node) => {
    if (node.node_type === "dataset") {
      const contextPath = ancestors.join(" / ");
      const summaryText = buildNodeSubtitleParts(node).join(" • ");
      return [
        {
          contextPath: contextPath || "Root",
          name: node.name,
          path: node.path,
          searchText: [node.name, node.path, contextPath, summaryText].join(" ").toLowerCase(),
          sectionLabel: buildDatasetSectionLabel(ancestors),
          summaryText,
        },
      ];
    }
    return flattenDatasets(node.children ?? [], [...ancestors, node.name]);
  });

export function Hdf5Navigator({ tree, selectedPath, onSelect, truncated }: Hdf5NavigatorProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [activePath, setActivePath] = useState<string | null>(null);
  const activePathRef = useRef<string | null>(null);
  const itemRefs = useRef(new Map<string, HTMLDivElement | null>());
  const datasetEntries = useMemo(() => flattenDatasets(tree), [tree]);
  const datasetCount = useMemo(() => countDatasets(tree), [tree]);
  const trimmedQuery = searchQuery.trim();

  const setActivePathValue = (path: string | null) => {
    activePathRef.current = path;
    setActivePath(path);
  };

  const displayedEntries = useMemo(
    () => filterDatasetEntries(datasetEntries, searchQuery, selectedPath),
    [datasetEntries, searchQuery, selectedPath]
  );

  const groupedEntries = useMemo(() => {
    const grouped = new Map<string, FlatDatasetEntry[]>();
    for (const entry of displayedEntries) {
      const existing = grouped.get(entry.sectionLabel);
      if (existing) {
        existing.push(entry);
      } else {
        grouped.set(entry.sectionLabel, [entry]);
      }
    }
    return Array.from(grouped.entries());
  }, [displayedEntries]);

  useEffect(() => {
    if (displayedEntries.length === 0) {
      setActivePathValue(null);
      return;
    }
    setActivePathValue(resolveActivePath(displayedEntries, activePathRef.current, selectedPath));
  }, [displayedEntries, selectedPath]);

  useEffect(() => {
    if (!activePath) {
      return;
    }
    const activeElement = itemRefs.current.get(activePath);
    activeElement?.scrollIntoView({ block: "nearest" });
  }, [activePath]);

  const moveActivePath = (direction: 1 | -1, queryValue: string) => {
    const entries = filterDatasetEntries(datasetEntries, queryValue, selectedPath);
    if (entries.length === 0) {
      return;
    }
    const currentIndex = entries.findIndex((entry) => entry.path === activePathRef.current);
    const nextIndex =
      currentIndex === -1
        ? direction === -1
          ? entries.length - 1
          : 0
        : (currentIndex + direction + entries.length) % entries.length;
    setActivePathValue(entries[nextIndex]?.path ?? activePathRef.current);
  };

  return (
    <Card
      className="viewer-hdf-dashboard-card viewer-hdf-navigator-card gap-0 py-0 shadow-none"
      data-hdf5-navigator="true"
    >
      <CardHeader className="viewer-hdf-dashboard-header viewer-hdf-dashboard-header-split">
        <div>
          <CardTitle>Dataset browser</CardTitle>
          <CardDescription>
            {trimmedQuery
              ? `${displayedEntries.length.toLocaleString()} matching dataset${displayedEntries.length === 1 ? "" : "s"}`
              : `${datasetCount.toLocaleString()} dataset${datasetCount === 1 ? "" : "s"}`}
          </CardDescription>
        </div>
        {truncated ? <p className="viewer-hdf-dashboard-note">Fast-open summary</p> : null}
      </CardHeader>

      <CardContent className="viewer-hdf-dashboard-content viewer-hdf-navigator-content">
        <Command className="viewer-hdf-command" shouldFilter={false} loop>
          <div className="viewer-hdf-command-toolbar" data-hdf5-search="true">
            <CommandInput
              value={searchQuery}
              onValueChange={(nextValue) => {
                setSearchQuery(nextValue);
                const nextEntries = filterDatasetEntries(datasetEntries, nextValue, selectedPath);
                setActivePathValue(resolveActivePath(nextEntries, activePathRef.current, selectedPath));
              }}
              placeholder="Find datasets by name, path, or type"
              aria-label="Find datasets"
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  moveActivePath(1, event.currentTarget.value);
                  return;
                }
                if (event.key === "ArrowUp") {
                  event.preventDefault();
                  moveActivePath(-1, event.currentTarget.value);
                  return;
                }
                if (event.key === "Enter") {
                  const currentEntries = filterDatasetEntries(
                    datasetEntries,
                    event.currentTarget.value,
                    selectedPath
                  );
                  const targetPath = resolveActivePath(
                    currentEntries,
                    activePathRef.current,
                    selectedPath
                  );
                  if (!targetPath) {
                    return;
                  }
                  event.preventDefault();
                  setActivePathValue(targetPath);
                  onSelect(targetPath);
                }
              }}
            />
            {trimmedQuery ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSearchQuery("");
                  setActivePathValue(resolveActivePath(datasetEntries, activePathRef.current, selectedPath));
                }}
              >
                <X className="size-4" />
                Clear
              </Button>
            ) : null}
          </div>

          <CommandList
            className="viewer-hdf-scroll-area viewer-hdf-tree-scroll viewer-hdf-search-results"
            data-hdf5-scroll-region="navigator"
            data-hdf5-dataset-list="true"
          >
            <CommandEmpty>No datasets matched this filter. Clear the search or choose a broader term.</CommandEmpty>
            {groupedEntries.map(([sectionLabel, entries]) => (
              <CommandGroup key={sectionLabel} heading={sectionLabel}>
                {entries.map((entry) => (
                  <CommandItem
                    key={entry.path}
                    value={entry.path}
                    keywords={[entry.name, entry.contextPath, entry.summaryText]}
                    onSelect={() => {
                      setActivePathValue(entry.path);
                      onSelect(entry.path);
                    }}
                    data-hdf5-path={entry.path}
                    ref={(element) => {
                      itemRefs.current.set(entry.path, element);
                    }}
                    className={cn(
                      "viewer-hdf-command-item",
                      selectedPath === entry.path && "is-selected",
                      activePath === entry.path && "is-active"
                    )}
                  >
                    <div className="viewer-hdf-command-item-copy">
                      <span className="viewer-hdf-node-name">{entry.name}</span>
                      {entry.summaryText ? <span className="viewer-hdf-node-meta">{entry.summaryText}</span> : null}
                      <span className="viewer-hdf-command-path">{entry.contextPath}</span>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            ))}
          </CommandList>
        </Command>
      </CardContent>
    </Card>
  );
}
