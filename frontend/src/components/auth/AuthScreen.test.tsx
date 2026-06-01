import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AuthScreen } from "./AuthScreen";

describe("AuthScreen", () => {
  it("uses an upbeat typewriter hero instead of account-connection copy", () => {
    const { container } = render(
      <AuthScreen
        bisqueRoot="https://bisque.example.org"
        loading={false}
        onAuthenticate={vi.fn()}
        onContinueGuest={vi.fn()}
      />
    );

    expect(screen.queryByText(/connect your bisque account/i)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /build the future/i })).toBeInTheDocument();

    expect(container.querySelector(".auth-hero-typewriter")).toBeInTheDocument();
    expect(container.querySelector(".auth-hero-typewriter-text")).toBeInTheDocument();
    expect(container.querySelector(".auth-hero-typewriter-caret")).toBeInTheDocument();
    expect(container.querySelector(".auth-hero-rotator")).not.toBeInTheDocument();
  });
});
