import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Sheet, SheetContent, SheetDescription, SheetTitle } from "./sheet";

describe("Sheet", () => {
  it("does not keep state slide transforms on anchored side sheets", () => {
    render(
      <Sheet open>
        <SheetContent side="right" data-testid="sheet-content">
          <SheetTitle>Resource filters</SheetTitle>
          <SheetDescription>Filter resources and choose a view.</SheetDescription>
        </SheetContent>
      </Sheet>
    );

    expect(screen.getByTestId("sheet-content").className).not.toContain(
      "data-[state=open]:slide-in-from-right"
    );
    expect(screen.getByTestId("sheet-content").className).not.toContain(
      "data-[state=closed]:slide-out-to-right"
    );
    expect(screen.getByTestId("sheet-content").className).toContain(
      "data-[state=closed]:hidden"
    );
  });
});
