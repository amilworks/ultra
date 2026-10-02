import type { BisqueSearchRequest, BisqueSearchResponse } from "../../types";

type SearchBisqueResources = (
  options: BisqueSearchRequest
) => Promise<BisqueSearchResponse>;

export const loadOptionalBisqueImageCount = async (
  searchBisqueResources: SearchBisqueResources
): Promise<number | null> => {
  try {
    const search = await searchBisqueResources({
      resourceType: "image",
      scope: "owner",
      limit: 1,
      countAll: true,
    });
    return Math.max(0, Number(search.count) || 0);
  } catch {
    return null;
  }
};
