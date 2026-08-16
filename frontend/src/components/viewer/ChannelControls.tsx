import { useId, useMemo, useRef, useState, type CSSProperties } from "react";
import { Check, ChevronDown, Search } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export const MAX_COMPOSITE_CHANNELS = 8;

const MAX_INLINE_CHANNELS = 12;
const CHANNEL_ROW_HEIGHT = 42;
const CHANNEL_BROWSER_HEIGHT = 320;
const CHANNEL_BROWSER_OVERSCAN = 2;

type ChannelControlsProps = {
  channelNames: string[];
  channelColors: string[];
  selectedIndices: number[];
  canEditColor: boolean;
  singleChannelMode: boolean;
  portalContainer?: HTMLElement | null;
  onToggleChannel: (index: number, active: boolean) => void;
  onSetChannelColor: (index: number, hex: string) => void;
};

type ChannelLabel = {
  name: string;
  meta: string;
};

const splitChannelLabel = (label: string): ChannelLabel => {
  const dash = label.lastIndexOf(" - ");
  return dash > 0
    ? { name: label.slice(0, dash), meta: label.slice(dash + 3) }
    : { name: label, meta: "" };
};

const ChannelChip = ({
  index,
  label,
  color,
  active,
  disabled,
  disabledDescriptionId,
  canEditColor,
  onToggleChannel,
  onSetChannelColor,
}: {
  index: number;
  label: string;
  color: string;
  active: boolean;
  disabled: boolean;
  disabledDescriptionId: string;
  canEditColor: boolean;
  onToggleChannel: ChannelControlsProps["onToggleChannel"];
  onSetChannelColor: ChannelControlsProps["onSetChannelColor"];
}) => {
  const { name, meta } = splitChannelLabel(label);
  return (
    <div
      className="viewer-channel-chip"
      data-active={active}
      data-disabled={disabled}
      data-viewer-channel-chip="true"
    >
      {canEditColor ? (
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="viewer-channel-swatch-btn"
              aria-label={`Edit ${name}, source channel ${index} color`}
              disabled={disabled}
            >
              <span
                className="viewer-channel-swatch"
                style={{ backgroundColor: color }}
                aria-hidden="true"
              />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            sideOffset={8}
            className="viewer-channel-color-popover"
          >
            <span className="viewer-channel-color-popover-label">{name}</span>
            <input
              type="color"
              aria-label={`${name}, source channel ${index} color`}
              value={color}
              onChange={(event) => onSetChannelColor(index, event.target.value)}
            />
          </PopoverContent>
        </Popover>
      ) : (
        <span
          className="viewer-channel-swatch viewer-channel-swatch-static"
          style={{ backgroundColor: color }}
          aria-hidden="true"
        />
      )}
      <button
        type="button"
        className="viewer-channel-toggle"
        aria-pressed={active}
        aria-label={`${name}, source channel ${index}`}
        aria-describedby={disabled ? disabledDescriptionId : undefined}
        disabled={disabled}
        title={
          disabled
            ? `Up to ${MAX_COMPOSITE_CHANNELS} channels can be combined`
            : label
        }
        onClick={() => onToggleChannel(index, active)}
      >
        <span className="viewer-channel-name">{name}</span>
        {meta ? <span className="viewer-channel-meta">{meta}</span> : null}
      </button>
    </div>
  );
};

export function ChannelControls({
  channelNames,
  channelColors,
  selectedIndices,
  canEditColor,
  singleChannelMode,
  portalContainer,
  onToggleChannel,
  onSetChannelColor,
}: ChannelControlsProps) {
  const [browserOpen, setBrowserOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [scrollTop, setScrollTop] = useState(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const capStatusId = useId();
  const selectedChannels = useMemo(
    () => {
      const normalized: number[] = [];
      const seen = new Set<number>();
      for (const index of selectedIndices) {
        if (
          !Number.isFinite(index) ||
          !Number.isInteger(index) ||
          index < 0 ||
          index >= channelNames.length ||
          seen.has(index)
        ) {
          continue;
        }
        normalized.push(index);
        seen.add(index);
        if (normalized.length === MAX_COMPOSITE_CHANNELS) {
          break;
        }
      }
      return normalized;
    },
    [channelNames.length, selectedIndices],
  );
  const selectedSet = useMemo(() => new Set(selectedChannels), [selectedChannels]);
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filteredChannels = useMemo(
    () =>
      channelNames
        .map((label, index) => ({ index, label }))
        .filter(({ index, label }) => {
          if (!normalizedQuery) {
            return true;
          }
          return `${label} ${index} ${index + 1} c${index}`
            .toLocaleLowerCase()
            .includes(normalizedQuery);
        }),
    [channelNames, normalizedQuery],
  );

  const resetBrowserScroll = () => {
    setScrollTop(0);
    if (viewportRef.current) {
      viewportRef.current.scrollTop = 0;
    }
  };

  const browserHeight = Math.min(
    CHANNEL_BROWSER_HEIGHT,
    Math.max(CHANNEL_ROW_HEIGHT, filteredChannels.length * CHANNEL_ROW_HEIGHT),
  );
  const mountedRowLimit = Math.ceil(browserHeight / CHANNEL_ROW_HEIGHT) +
    CHANNEL_BROWSER_OVERSCAN * 2;
  const firstVisibleIndex = Math.min(
    Math.max(0, filteredChannels.length - mountedRowLimit),
    Math.max(
      0,
      Math.floor(scrollTop / CHANNEL_ROW_HEIGHT) - CHANNEL_BROWSER_OVERSCAN,
    ),
  );
  const lastVisibleIndex = Math.min(
    filteredChannels.length,
    firstVisibleIndex + mountedRowLimit,
  );
  const visibleChannels = filteredChannels.slice(firstVisibleIndex, lastVisibleIndex);
  const atCompositeLimit =
    !singleChannelMode && selectedChannels.length >= MAX_COMPOSITE_CHANNELS;
  const capExplanation = atCompositeLimit
    ? `Maximum ${MAX_COMPOSITE_CHANNELS} channels selected. Remove a selected channel to choose another.`
    : "";

  if (channelNames.length <= MAX_INLINE_CHANNELS) {
    return (
      <div
        className="viewer-channel-controls"
        data-viewer-channel-controls="true"
        role="group"
        aria-label="Channels"
      >
        {channelNames.map((label, index) => (
          <ChannelChip
            key={`${label}-${index}`}
            index={index}
            label={label}
            color={channelColors[index]}
            active={selectedSet.has(index)}
            disabled={atCompositeLimit && !selectedSet.has(index)}
            disabledDescriptionId={capStatusId}
            canEditColor={canEditColor}
            onToggleChannel={onToggleChannel}
            onSetChannelColor={onSetChannelColor}
          />
        ))}
        <span id={capStatusId} className="sr-only" role="status" aria-live="polite">
          {capExplanation}
        </span>
      </div>
    );
  }

  return (
    <div
      className="viewer-channel-controls viewer-channel-controls-bounded"
      data-viewer-channel-controls="true"
      role="group"
      aria-label="Channels"
    >
      {selectedChannels.map((index) => (
        <ChannelChip
          key={`${channelNames[index]}-${index}`}
          index={index}
          label={channelNames[index]}
          color={channelColors[index]}
          active
          disabled={false}
          disabledDescriptionId={capStatusId}
          canEditColor={canEditColor}
          onToggleChannel={onToggleChannel}
          onSetChannelColor={onSetChannelColor}
        />
      ))}
      <Dialog
        open={browserOpen}
        onOpenChange={(open) => {
          setBrowserOpen(open);
          if (!open) {
            setQuery("");
            resetBrowserScroll();
          }
        }}
      >
        <DialogTrigger asChild>
          <button
            type="button"
            className="viewer-channel-browser-trigger"
            aria-label={`Choose channels, ${selectedChannels.length} selected of ${channelNames.length}`}
          >
            <span>Channels</span>
            <span className="viewer-channel-browser-count">
              {selectedChannels.length} / {channelNames.length}
            </span>
            <ChevronDown aria-hidden="true" />
          </button>
        </DialogTrigger>
        <DialogContent
          portalContainer={portalContainer}
          className="viewer-channel-browser-dialog"
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            searchRef.current?.focus();
          }}
        >
          <DialogHeader className="viewer-channel-browser-head">
            <div>
              <DialogTitle>Channels</DialogTitle>
              <DialogDescription>
                {singleChannelMode
                  ? "Choose one signal to view."
                  : `Combine up to ${MAX_COMPOSITE_CHANNELS} signals.`}
              </DialogDescription>
            </div>
            <span className="viewer-channel-browser-total">
              {channelNames.length} total
            </span>
          </DialogHeader>
          <label className="viewer-channel-search">
            <Search aria-hidden="true" />
            <Input
              ref={searchRef}
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                resetBrowserScroll();
              }}
              placeholder={`Search ${channelNames.length} channels`}
              aria-label="Search channels"
              autoComplete="off"
            />
          </label>
          <div
            ref={viewportRef}
            className="viewer-channel-browser-viewport"
            style={{
              height: browserHeight,
              "--viewer-channel-row-height": `${CHANNEL_ROW_HEIGHT}px`,
            } as CSSProperties}
            aria-label="Channel catalog"
            tabIndex={0}
            onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
          >
            {filteredChannels.length > 0 ? (
              <div
                className="viewer-channel-browser-spacer"
                style={{ height: filteredChannels.length * CHANNEL_ROW_HEIGHT }}
              >
                {visibleChannels.map(({ index, label }, visibleOffset) => {
                  const active = selectedSet.has(index);
                  const disabled = atCompositeLimit && !active;
                  const { name, meta } = splitChannelLabel(label);
                  return (
                    <div
                      key={`${label}-${index}`}
                      className="viewer-channel-browser-row"
                      data-active={active}
                      data-disabled={disabled}
                      data-viewer-channel-row="true"
                      style={{
                        transform: `translateY(${(firstVisibleIndex + visibleOffset) * CHANNEL_ROW_HEIGHT}px)`,
                      }}
                    >
                      {canEditColor ? (
                        <input
                          type="color"
                          className="viewer-channel-browser-color"
                          aria-label={`Edit ${name}, source channel ${index} color`}
                          value={channelColors[index]}
                          disabled={disabled}
                          onChange={(event) => onSetChannelColor(index, event.target.value)}
                        />
                      ) : (
                        <span
                          className="viewer-channel-swatch viewer-channel-browser-swatch"
                          style={{ backgroundColor: channelColors[index] }}
                          aria-hidden="true"
                        />
                      )}
                      <button
                        type="button"
                        aria-label={`${name}, source channel ${index}`}
                        aria-pressed={active}
                        aria-describedby={disabled ? capStatusId : undefined}
                        disabled={disabled}
                        title={
                          disabled
                            ? `Up to ${MAX_COMPOSITE_CHANNELS} channels can be combined`
                            : label
                        }
                        onClick={() => onToggleChannel(index, active)}
                      >
                        <span className="viewer-channel-browser-label">
                          <span>{name}</span>
                          {meta ? <small>{meta}</small> : null}
                        </span>
                        <span className="viewer-channel-browser-index">
                          C{index}
                        </span>
                        {active ? <Check aria-hidden="true" /> : null}
                      </button>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="viewer-channel-browser-empty" role="status">
                No matching channels
              </div>
            )}
          </div>
          <div className="viewer-channel-browser-foot" aria-live="polite">
            <span>
              {filteredChannels.length} matching, {selectedChannels.length} selected
            </span>
            {atCompositeLimit ? (
              <span>Remove a channel to choose another.</span>
            ) : null}
          </div>
          <span id={capStatusId} className="sr-only" role="status" aria-live="polite">
            {capExplanation}
          </span>
        </DialogContent>
      </Dialog>
    </div>
  );
}
