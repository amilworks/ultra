from ultra_deepagents.rarespot.tiling import (
    build_sliding_tiles,
    classwise_nms,
    remap_tile_box,
)


def test_sliding_tiles_use_25_percent_overlap_and_include_edges():
    tiles = build_sliding_tiles(width=250, height=120, tile_size=100, overlap=0.25)

    assert tiles.stride == 75
    assert [(tile.x0, tile.y0, tile.x1, tile.y1) for tile in tiles.tiles] == [
        (0, 0, 100, 100),
        (75, 0, 175, 100),
        (150, 0, 250, 100),
        (0, 20, 100, 120),
        (75, 20, 175, 120),
        (150, 20, 250, 120),
    ]


def test_tile_box_remaps_to_image_coordinates():
    remapped = remap_tile_box(
        [10.0, 20.0, 40.0, 70.0],
        tile_x0=75,
        tile_y0=20,
        image_width=250,
        image_height=120,
    )

    assert remapped == [85.0, 40.0, 115.0, 90.0]


def test_classwise_nms_keeps_overlapping_boxes_from_different_classes():
    rows = [
        {
            "class_id": 0,
            "class_name": "prairie_dog",
            "confidence": 0.90,
            "xyxy": [0.0, 0.0, 100.0, 100.0],
        },
        {
            "class_id": 0,
            "class_name": "prairie_dog",
            "confidence": 0.50,
            "xyxy": [5.0, 5.0, 95.0, 95.0],
        },
        {
            "class_id": 1,
            "class_name": "burrow",
            "confidence": 0.80,
            "xyxy": [5.0, 5.0, 95.0, 95.0],
        },
    ]

    kept = classwise_nms(rows, iou_threshold=0.45)

    assert [(row["class_name"], row["confidence"]) for row in kept] == [
        ("prairie_dog", 0.90),
        ("burrow", 0.80),
    ]
