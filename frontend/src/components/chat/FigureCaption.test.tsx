import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ApiClient } from "@/lib/api";

import { FigureCaption } from "./FigureCaption";

// jsdom has no IntersectionObserver, so the component takes its immediate-fetch
// fallback path — ideal for asserting the rendered caption.

function client(overrides: Partial<ApiClient>): ApiClient {
  return overrides as unknown as ApiClient;
}

describe("FigureCaption", () => {
  it("renders a calm caption after lazily fetching it", async () => {
    const getRunArtifactCaption = vi
      .fn()
      .mockResolvedValue({ caption: "A scatter plot of confidence versus stability for two classes.", enabled: true });
    render(
      <FigureCaption
        runId="run-1"
        path="outputs/fig.png"
        apiClient={client({ getRunArtifactCaption })}
      />
    );
    await waitFor(() =>
      expect(screen.getByText(/scatter plot of confidence versus stability/i)).toBeInTheDocument()
    );
    expect(screen.getByText("Figure.")).toBeInTheDocument();
    expect(getRunArtifactCaption).toHaveBeenCalledWith("run-1", "outputs/fig.png");
  });

  it("renders nothing when captioning is disabled", async () => {
    const getRunArtifactCaption = vi.fn().mockResolvedValue({ caption: "", enabled: false });
    const { container } = render(
      <FigureCaption runId="run-2" path="outputs/fig2.png" apiClient={client({ getRunArtifactCaption })} />
    );
    await waitFor(() => expect(getRunArtifactCaption).toHaveBeenCalled());
    expect(container.querySelector(".chat-figure-caption-label")).toBeNull();
    // The empty sentinel stays (for the observer) but carries no text.
    expect(container.querySelector(".chat-figure-caption-empty")).not.toBeNull();
  });

  it("does not fetch without a runId", () => {
    const getRunArtifactCaption = vi.fn();
    render(<FigureCaption runId={null} path="outputs/fig.png" apiClient={client({ getRunArtifactCaption })} />);
    expect(getRunArtifactCaption).not.toHaveBeenCalled();
  });
});
