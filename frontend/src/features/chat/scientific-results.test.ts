import { describe, expect, it } from "vitest";

import {
  buildScientificResultGroups,
  type ScientificResultArtifact,
  type ScientificResultProgressEvent,
  type ScientificToolInvocation,
} from "./scientific-results";

describe("buildScientificResultGroups", () => {
  it("collapses duplicate megaseg runs into one result surface and merges quantification metrics", () => {
    const progressEvents: ScientificResultProgressEvent[] = [
      {
        event: "completed",
        tool: "segment_image_megaseg",
        summary: {
          success: true,
          result_group_id: "megaseg-group-1",
          output_directory: "/tmp/megaseg-run-a",
        },
      },
      {
        event: "completed",
        tool: "segment_image_megaseg",
        summary: {
          success: true,
          result_group_id: "megaseg-group-1",
          output_directory: "/tmp/megaseg-run-b",
        },
      },
      {
        event: "completed",
        tool: "quantify_segmentation_masks",
        summary: {
          success: true,
          result_group_id: "megaseg-group-1",
          mean_coverage_percent: 0.54306,
          mean_object_count: 112,
        },
      },
    ];

    const toolInvocations: ScientificToolInvocation[] = [
      {
        tool: "segment_image_megaseg",
        status: "completed",
        output_envelope: {
          success: true,
          result_group_id: "megaseg-group-1",
          report_path: "/tmp/megaseg_report.md",
          summary_csv_path: "/tmp/megaseg_summary.csv",
          visualization_paths: [
            {
              kind: "overlay_mip",
              title: "Megaseg overlay (MIP)",
              path: "tool_outputs/overlay_mip.png",
              file: "NPM1_13054_IM.tiff",
            },
            {
              kind: "overlay_mid_z",
              title: "Megaseg overlay (mid Z)",
              path: "tool_outputs/overlay_midz.png",
              file: "NPM1_13054_IM.tiff",
            },
            {
              kind: "mask_preview",
              title: "Binary mask preview",
              path: "tool_outputs/mask_preview.png",
              file: "NPM1_13054_IM.tiff",
            },
          ],
          scientific_summary: {
            files: [
              {
                "file": "NPM1_13054_IM.tiff",
                "coverage_percent": 0.54306,
                "object_count": 112,
                "active_slice_count": 25,
                "z_slice_count": 65,
                "largest_component_voxels": 13615,
                "technical_summary": "Sparse punctate segmentation within nuclei.",
              },
            ],
          },
        },
      },
      {
        tool: "quantify_segmentation_masks",
        status: "completed",
        output_envelope: {
          success: true,
          result_group_id: "megaseg-group-1",
          summary: {
            "mean_coverage_percent": 0.54306,
            "mean_object_count": 112,
          },
          rows: [
            {
              "mask_path": "/tmp/NPM1_13054_IM__megaseg_mask.tiff",
              "coverage_percent": 0.54306,
              "object_count": 112,
            },
          ],
        },
      },
    ];

    const runArtifacts: ScientificResultArtifact[] = [
      {
        path: "tool_outputs/overlay_mip.png",
        title: "Megaseg overlay (MIP)",
        sourcePath: "NPM1_13054_IM.tiff",
        url: "/artifacts/overlay_mip.png",
        downloadUrl: "/artifacts/overlay_mip.png",
        previewable: true,
        resultGroupId: "megaseg-group-1",
      },
      {
        path: "tool_outputs/overlay_midz.png",
        title: "Megaseg overlay (mid Z)",
        sourcePath: "NPM1_13054_IM.tiff",
        url: "/artifacts/overlay_midz.png",
        downloadUrl: "/artifacts/overlay_midz.png",
        previewable: true,
        resultGroupId: "megaseg-group-1",
      },
      {
        path: "tool_outputs/mask_preview.png",
        title: "Binary mask preview",
        sourcePath: "NPM1_13054_IM.tiff",
        url: "/artifacts/mask_preview.png",
        downloadUrl: "/artifacts/mask_preview.png",
        previewable: true,
        resultGroupId: "megaseg-group-1",
      },
    ];

    const groups = buildScientificResultGroups({
      progressEvents,
      toolInvocations,
      runArtifacts,
    });

    expect(groups).toHaveLength(1);
    expect(groups[0].resultGroupId).toBe("megaseg-group-1");
    expect(groups[0].metrics.coveragePercent).toBeCloseTo(0.54306, 6);
    expect(groups[0].metrics.objectCount).toBe(112);
    expect(groups[0].metrics.activeSliceCount).toBe(25);
    expect(groups[0].metrics.zSliceCount).toBe(65);
    expect(groups[0].metrics.largestComponentVoxels).toBe(13615);
    expect(groups[0].reportPath).toBe("/tmp/megaseg_report.md");
    expect(groups[0].summaryCsvPath).toBe("/tmp/megaseg_summary.csv");
    expect(groups[0].heroFigure?.kind).toBe("overlay_mip");
    expect(groups[0].secondaryFigures.map((item) => item.kind)).toEqual([
      "overlay_mid_z",
      "mask_preview",
    ]);
  });
});
