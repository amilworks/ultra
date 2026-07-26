import { describe, expect, it } from "vitest";

import {
  planePointToTextureRatios,
  textureRatiosToPlanePoint,
} from "./SlicePlaneCanvas";

describe("slice plane delivery coordinates", () => {
  it("places native-grid first and last voxels at texel centres", () => {
    const grid = { width: 4, height: 3 };
    expect(planePointToTextureRatios({ row: 0, col: 0 }, grid)).toEqual({
      x: 0.125,
      y: 1 / 6,
    });
    expect(planePointToTextureRatios({ row: 2, col: 3 }, grid)).toEqual({
      x: 0.875,
      y: 5 / 6,
    });
  });

  it("round-trips every delivery texel centre through click mapping", () => {
    const grid = { width: 7, height: 5 };
    for (let row = 0; row < grid.height; row += 1) {
      for (let col = 0; col < grid.width; col += 1) {
        const point = { row, col };
        expect(
          textureRatiosToPlanePoint(planePointToTextureRatios(point, grid), grid)
        ).toEqual(point);
      }
    }
  });
});
