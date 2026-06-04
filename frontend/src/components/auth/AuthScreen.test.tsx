import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AuthScreen } from "./AuthScreen";

describe("AuthScreen", () => {
  it("uses an upbeat typewriter hero instead of account-connection copy", () => {
    const { container } = render(
      <AuthScreen
        bisqueRoot="https://bisque.example.org"
        loading={false}
        onAuthenticate={vi.fn()}
        onRequestAccount={vi.fn()}
      />
    );

    expect(screen.queryByText(/connect your bisque account/i)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /build the future/i })).toBeInTheDocument();

    expect(container.querySelector(".auth-hero-typewriter")).toBeInTheDocument();
    expect(container.querySelector(".auth-hero-typewriter-text")).toBeInTheDocument();
    expect(container.querySelector(".auth-hero-typewriter-caret")).toBeInTheDocument();
    expect(container.querySelector(".auth-hero-rotator")).not.toBeInTheDocument();
  });

  it("uses a hosted WorkOS sign-in surface without credential fields", () => {
    render(
      <AuthScreen
        authProvider="workos"
        bisqueRoot="https://bisque.example.org"
        loading={false}
        onAuthenticate={vi.fn()}
        onRequestAccount={vi.fn()}
        onStartHostedAuth={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: /sign in with workos/i })).toBeInTheDocument();
    expect(screen.queryByLabelText(/username/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/password/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /continue as guest/i })).not.toBeInTheDocument();
  });

  it("presents local guest intake as an account request", () => {
    render(
      <AuthScreen
        bisqueRoot="https://bisque.example.org"
        loading={false}
        onAuthenticate={vi.fn()}
        onRequestAccount={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: "Request an Account" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /continue as guest/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Request an Account" }));

    expect(screen.getAllByText("Request an Account").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/administrator can review access/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/affiliation/i)).toBeInTheDocument();
  });

  it("submits account requests instead of creating a guest session", () => {
    const onRequestAccount = vi.fn();
    render(
      <AuthScreen
        bisqueRoot="https://bisque.example.org"
        loading={false}
        onAuthenticate={vi.fn()}
        onRequestAccount={onRequestAccount}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Request an Account" }));
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "Ada Lovelace" },
    });
    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "ada@example.org" },
    });
    fireEvent.change(screen.getByLabelText("Affiliation"), {
      target: { value: "Analytical Engine Lab" },
    });
    const requestButtons = screen.getAllByRole("button", { name: "Request an Account" });
    fireEvent.click(requestButtons[requestButtons.length - 1]);

    expect(onRequestAccount).toHaveBeenCalledWith({
      name: "Ada Lovelace",
      email: "ada@example.org",
      affiliation: "Analytical Engine Lab",
    });
  });

  it("shows account approval status messages without marking them as form errors", () => {
    const { container } = render(
      <AuthScreen
        bisqueRoot="https://bisque.example.org"
        loading={false}
        statusMessage="Your account request is pending administrator approval."
        onAuthenticate={vi.fn()}
        onRequestAccount={vi.fn()}
      />
    );

    expect(screen.getByText(/pending administrator approval/i)).toBeInTheDocument();
    expect(container.querySelector(".auth-status-message")).toBeInTheDocument();
    expect(container.querySelector(".auth-error")).not.toBeInTheDocument();
  });
});
