+++
pack_id = "ecology.prairie_rare_spot"
title = "RareSpot prairie dog guidance"
scope = "domain"
domains = ["ecology"]
workflow_ids = ["detect_prairie_dog"]
keywords = ["prairie dog", "prairie dogs", "burrow", "burrows", "wildlife", "conservation", "keystone species", "habitat", "rarespot", "aerial survey", "orthomosaic"]
priority = 24
max_chars = 2600
+++

# RareSpot Prairie Dog Guidance

Use this reference for prairie dog, burrow, wildlife monitoring, and habitat-context turns.

## Ecological framing
- Prairie dogs are keystone species, so detections should be framed as ecologically meaningful survey evidence.
- Treat image detections as observations from aerial surveys, not as a population census unless the sampling design supports that claim.
- Prefer careful language such as "visible", "consistent with", and "survey suggests" when interpreting outputs.
- Burrow proximity is an important ecological signal, but it should be reported as image-space distance unless georeferencing or pixel scale is known.
- If the current artifact is a single tile or crop, keep the interpretation local to the visible area rather than extrapolating to the full colony or site.

## Survey context from the paper
- The paper uses fixed-wing drone imagery over grassland habitat at about 100 m altitude and 2 cm/pixel resolution.
- Orthomosaics were generated from flights with about 70% overlap between frames.
- The inference and annotation workflow uses tiled imagery; the supplementary material describes 512x512 patches with 25% overlap.
- Orthomosaics are ecologically important because they reduce duplicate counts across overlapping raw frames and provide a georeferenced survey surface.
- Prairie dog boxes in the paper are small and low contrast, so weakly separated detections should be interpreted cautiously.
- Reported object sizes are small: prairie dogs are roughly 11-98 px wide (mean about 33.5 px), while burrows are somewhat larger and more variable.

## Common confounders
- Shadows, rocks, vegetation, dirt, border effects, and other natural clutter can look like prairie dogs or burrows.
- Burrow entrances are especially easy to confuse with shadows or rocks.
- The supplementary figures also highlight false positives from sticks, long shadows, and bush shadows, plus missed prairie dogs near tile borders.
- Aerial imagery is vulnerable to occlusion and scale effects, so false negatives are expected in dense or grassy scenes.

## Useful quantitative outputs
- Counts by class: `prairie_dog` and `burrow`.
- Overlap between prairie dogs and burrows.
- Nearest burrow distance per prairie dog and summary statistics across images.
- Metadata context when available: capture time, GPS latitude/longitude, image dimensions, and tile overlap.
- If the image is georeferenced, note that survey date, site identity, orthomosaic vs tile status, and duplicate-merge assumptions materially change ecological interpretation.

## Detection reliability from the paper
- On the held-out colony test set, the RareSpot model improved prairie dog mAP@50 to about 0.495 and prairie dog recall to about 0.519, while burrow mAP@50 reached about 0.923.
- Burrows are more prevalent and generally easier to detect than prairie dogs, so absence of burrow boxes alongside prairie dog detections can still reflect visibility limits rather than true absence.

## Preferred language
- Use "ecology and wildlife monitoring" rather than generic computer-vision framing when the prompt is about prairie dogs.
- Use "keystone-species monitoring", "habitat context", "burrow proximity", and "survey evidence".
- Avoid overclaiming that detections alone prove colony health, abundance, or occupancy.

## What to emphasize in responses
- The image scale, tile overlap, and low-contrast nature of prairie dog detection.
- Whether detections are near burrows or isolated from them.
- Whether the imagery is orthomosaic-based and therefore suited to survey-level interpretation.
- That the RareSpot pipeline was designed to improve small-object recall and reduce confusion from natural background textures.
