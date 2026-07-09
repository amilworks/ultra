import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown } from "./markdown";

const cases: Array<{ name: string; md: string; expectTable: boolean; lens: string }> = [
  { name: "thematic_break_after_pipe_line", md: "Choose option A | B | C below\n---\n\nThe rule takes effect immediately.", expectTable: false, lens: "false-positive" },
  { name: "setext_h2_with_pipe_in_heading", md: "Pros | Cons of the approach\n---\n\nWe weighed both sides carefully.", expectTable: false, lens: "false-positive" },
  { name: "prose_pipes_and_dashes", md: "Use grep | sort to filter, then run the build --- it just works.", expectTable: false, lens: "false-positive" },
  { name: "list_items_with_pipes", md: "- alpha | beta\n- gamma | delta\n- epsilon | zeta", expectTable: false, lens: "false-positive" },
  { name: "inline_code_pipe_delimiter", md: "The pattern `|---|---|` denotes a two-column delimiter row in GFM.", expectTable: false, lens: "false-positive" },
  { name: "paragraph_describing_table_syntax", md: "To make a table, write the header, then a line like\n|---|---|\nand the rows follow.", expectTable: false, lens: "false-positive" },
  { name: "raw_html_table", md: "<table>\n<tr><td>a</td><td>b</td></tr>\n</table>", expectTable: false, lens: "false-positive" },
  { name: "hr_between_pipe_paragraphs", md: "Latency is 5ms | throughput 10k rps.\n\n---\n\nMemory | CPU stayed flat all week.", expectTable: false, lens: "false-positive" },
  { name: "align_colons_lcr", md: "| Left | Center | Right |\n| :--- | :----: | ----: |\n| a | b | c |", expectTable: true, lens: "valid-preserve" },
  { name: "no_outer_pipes", md: "a | b\n--- | ---\n1 | 2", expectTable: true, lens: "valid-preserve" },
  { name: "ragged_data_fewer_and_more", md: "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 |\n| 3 | 4 | 5 |", expectTable: true, lens: "valid-preserve" },
  { name: "rich_cell_content", md: "| Name | Detail |\n| --- | --- |\n| **bold** | `code` |\n| [link](http://x) | $a+b$ |", expectTable: true, lens: "valid-preserve" },
  { name: "wide_eight_column", md: "| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |", expectTable: true, lens: "valid-preserve" },
  { name: "escaped_pipe_in_cells", md: "| Op | Meaning |\n| --- | --- |\n| a \\| b | bitwise or |\n| x \\| y \\| z | chain |", expectTable: true, lens: "valid-preserve" },
  { name: "header_only_no_data", md: "| Col A | Col B |\n| --- | --- |", expectTable: true, lens: "valid-preserve" },
  { name: "unescaped_pipe_in_inline_math_data", md: "| Quantity | Formula |\n| --- | --- |\n| magnitude | $|x|$ |", expectTable: true, lens: "valid-preserve" },
  { name: "delim_one_more_col", md: "| A | B |\n| --- | --- | --- |\n| 1 | 2 |", expectTable: true, lens: "llm-malformed" },
  { name: "delim_one_fewer_col", md: "| Name | Age | City |\n| --- | --- |\n| Al | 30 | LA |", expectTable: true, lens: "llm-malformed" },
  { name: "inconsistent_spacing_extra_align_cell", md: "| Metric | Value |\n| :--- | ---: | :---: |\n| Loss | 0.12 |", expectTable: true, lens: "llm-malformed" },
  { name: "equals_underline_delim", md: "| Col A | Col B |\n| ==== | ==== |\n| x | y |", expectTable: false, lens: "llm-malformed" },
  { name: "unicode_box_dash_delim", md: "| Col A | Col B |\n| \u2500\u2500\u2500 | \u2500\u2500\u2500 |\n| x | y |", expectTable: false, lens: "llm-malformed" },
  { name: "header_data_agree_delim_single", md: "| Model | mAP |\n| --- |\n| YOLO | 0.78 |\n| RCNN | 0.71 |", expectTable: true, lens: "llm-malformed" },
  { name: "colons_wrong_count_drops_align", md: "| Left | Right |\n| :-- | :-: | --: |\n| a | b |", expectTable: true, lens: "llm-malformed" },
  { name: "double_trailing_pipe_empty_cell", md: "| P | Q |\n| --- | --- ||\n| 1 | 2 |", expectTable: false, lens: "llm-malformed" },
  { name: "escaped_pipe_in_header", md: "| a \\| b | c |\n| --- | --- | --- |\n| 1 | 2 |", expectTable: true, lens: "llm-malformed" },
  { name: "interior_colon_delim_cell", md: "| A | B |\n| :-:- | --- |\n| 1 | 2 |", expectTable: false, lens: "llm-malformed" },
  { name: "abs-delta-sigma-3col", md: "| Estimate | |\u0394|/\u03c3 | Class |\n| --- | --- | --- |\n| GR86 | 3.4 | high |\n| BRZ | 0.9 | low |", expectTable: false, lens: "ambiguous" },
  { name: "data-row-pipe-inflates", md: "| Metric | Value |\n| --- | --- |\n| range | 3|5 |\n| gap | 2|7 |", expectTable: true, lens: "ambiguous" }, // header==delim: pure GFM-native render, repair is a no-op
  { name: "header-pipe-alignment-preserved", md: "| Cmd | a|b | out |\n| :-- | --: |\n| ls | x | y |", expectTable: false, lens: "ambiguous" },
  { name: "pq-arm-header-vs-data", md: "| Gene | p|q arm | count |\n| --- | --- |\n| TP53 | p | 12 |\n| MYC | q | 8 |", expectTable: false, lens: "ambiguous" },
  { name: "modal-tiebreak-fires-wrongly", md: "| Signal | S/N | verdict |\n| --- | --- |\n| src1 | 5 | keep |\n| src2 | |z|>2 | drop |", expectTable: false, lens: "ambiguous" },
  { name: "no-data-rows-fires-wrongly", md: "| Estimate | |\u0394|/\u03c3 | Class |\n| --- | --- |", expectTable: false, lens: "ambiguous" },
  { name: "header-and-data-share-stray-pipe", md: "| a | b|c | d |\n| --- | --- |\n| 1 | 2|3 | 4 |", expectTable: true, lens: "ambiguous" }, // header+data both 4-consistent: repair matches GFM semantics for a correct delimiter
  { name: "data-pipe-header-matches-delim", md: "| Key | Val |\n| --- | --- | --- |\n| lo | a|b |\n| hi | c|d |", expectTable: false, lens: "ambiguous" },
  { name: "blockquote-bad-delim-stays-raw", md: "> | Name | Role | Team |\n> | --- | --- |\n> | Ann | Lead | Ops |", expectTable: false, lens: "structural-nesting" },
  { name: "blockquote-good-delim-renders", md: "> | Name | Role |\n> | --- | --- |\n> | Ann | Lead |", expectTable: true, lens: "structural-nesting" },
  { name: "list-item-table-bad-delim", md: "- | A | B | C |\n  | --- | --- |\n  | 1 | 2 | 3 |", expectTable: false, lens: "structural-nesting" },
  { name: "two-tables-one-valid-one-bad", md: "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n| C | D | E |\n| --- | --- |\n| 3 | 4 | 5 |", expectTable: true, lens: "structural-nesting" },
  { name: "table-after-heading-no-blank", md: "## Results\n| A | B |\n| --- |\n| 1 | 2 |", expectTable: true, lens: "structural-nesting" },
  { name: "table-adjacent-display-math", md: "$$\na - b\n$$\n\n| A | B | C |\n| --- | --- |\n| 1 | 2 | 3 |", expectTable: true, lens: "structural-nesting" },
  { name: "table-data-row-after-blank-line", md: "| A | B | C |\n| --- | --- |\n| 1 | 2 | 3 |\n\n| 4 | 5 | 6 |", expectTable: true, lens: "structural-nesting" },
  { name: "crlf-line-endings", md: "| A | B | C |\r\n| --- | --- |\r\n| 1 | 2 | 3 |", expectTable: true, lens: "structural-nesting" },
  { name: "greek_delta_sigma_2col_delim", md: "| Parameter | \u0394 (mean) | \u03c3 |\n| --- | --- |\n| learning rate | 3.0\u00d710\u207b\u2074 | 0.12 |\n| \u0393 decay | 0.961 | 0.03 |", expectTable: true, lens: "unicode-scientific" },
  { name: "subscripts_over_by_one_4col_delim", md: "| Layer | n_params | dtype |\n|---|---|---|---|\n| conv\u2081 | 1.2M | float32 |\n| conv\u2082 | 3.4M | float32 |", expectTable: true, lens: "unicode-scientific" },
  { name: "scinot_alignment_preserved", md: "| Metric | Value | Unit |\n| :-- | --: |\n| \u03b5 (tol) | ~1\u00d710\u207b\u2077 | \u2014 |\n| flux \u03a6 | 6.6\u00d710\u207b\u00b3 | W\u00b7m\u207b\u00b2 |", expectTable: true, lens: "unicode-scientific" },
  { name: "iou_single_cell_delim", md: "| Model | IoU | mAP\u2085\u2080 |\n|-------|\n| baseline | 0.812 | 0.785 |\n| ours | 0.961 | 0.842 |", expectTable: true, lens: "unicode-scientific" },
  { name: "unescaped_pipe_in_header_ambiguous", md: "| Range (a | b) | \u0394\u03c3 |\n| --- |\n| [0, 1] | 0.05 |\n| [1, 2] | 0.07 |", expectTable: false, lens: "unicode-scientific" },
  { name: "escaped_pipe_header_contrast", md: "| Range (a \\| b) | \u0394\u03c3 |\n| --- |\n| [0, 1] | 0.05 |\n| [1, 2] | 0.07 |", expectTable: true, lens: "unicode-scientific" },
  { name: "greek_five_col_undercount", md: "| \u03b8 | \u03c6 | \u03c8 | \u03c7 | \u03c9 |\n| --- | --- |\n| 1 | 2 | 3 | 4 | 5 |\n| 6 | 7 | 8 | 9 | 10 |", expectTable: true, lens: "unicode-scientific" },
  { name: "emdash_ge_overcount_data_agree", md: "| Condition | Threshold | Count |\n| --- | --- | --- | --- |\n| \u03c3 \u2265 3\u00d7 | \u22651\u00d710\u207b\u2077 | 42 |\n| \u0394 \u2264 0.5 | \u2014 | 17 |", expectTable: true, lens: "unicode-scientific" },
];

describe("Markdown table repair — adversarial matrix (50 generated cases)", () => {
  for (const c of cases) {
    it(`${c.lens} :: ${c.name}`, () => {
      const { container } = render(<Markdown>{c.md}</Markdown>);
      const hasTable = container.querySelector("table") !== null;
      expect(hasTable, `md=${JSON.stringify(c.md)}`).toBe(c.expectTable);
    });
  }
});
