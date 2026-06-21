import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ErrorBoundary } from "./ErrorBoundary";

function Boom({ explode }: { explode: boolean }) {
  if (explode) {
    throw new Error("kaboom");
  }
  return <div>healthy content</div>;
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    // The boundary reports caught errors; keep test output clean.
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders children when nothing throws", () => {
    render(
      <ErrorBoundary>
        <Boom explode={false} />
      </ErrorBoundary>
    );
    expect(screen.getByText("healthy content")).toBeInTheDocument();
  });

  it("renders the calm inline fallback when a child throws", () => {
    render(
      <ErrorBoundary>
        <Boom explode />
      </ErrorBoundary>
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("This part couldn’t be displayed.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("supports a custom fallback receiving the error", () => {
    render(
      <ErrorBoundary fallback={({ error }) => <p>custom: {error.message}</p>}>
        <Boom explode />
      </ErrorBoundary>
    );
    expect(screen.getByText("custom: kaboom")).toBeInTheDocument();
  });

  it("auto-recovers when resetKeys change", () => {
    function Harness() {
      const [explode, setExplode] = useState(true);
      return (
        <>
          <button type="button" onClick={() => setExplode(false)}>
            fix it
          </button>
          <ErrorBoundary resetKeys={[explode]}>
            <Boom explode={explode} />
          </ErrorBoundary>
        </>
      );
    }

    render(<Harness />);
    expect(screen.getByRole("alert")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "fix it" }));

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("healthy content")).toBeInTheDocument();
  });
});
