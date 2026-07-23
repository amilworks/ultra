import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BrandWordmark } from "./BrandWordmark";

describe("BrandWordmark", () => {
  it("exposes one accessible brand name while keeping the visual weight split", () => {
    render(<BrandWordmark className="test-wordmark" />);

    const wordmark = screen.getByRole("img", { name: "BisQue Ultra" });
    expect(wordmark).toHaveClass("brand-wordmark", "test-wordmark");
    expect(wordmark.querySelector(".brand-wordmark__bisque")).toHaveTextContent("BisQue");
    expect(wordmark.querySelector(".brand-wordmark__ultra")).toHaveTextContent("Ultra");
    expect(wordmark.querySelectorAll('[aria-hidden="true"]')).toHaveLength(2);
  });
});
