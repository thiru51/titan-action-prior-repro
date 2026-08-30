from titan.data import schema
from titan.data.common import denormalize_boxes, ltwh_to_cxcywh, normalize_boxes
from titan.paper import PAPER_RESULTS

import torch


def test_column_aliases_resolve_both_spellings():
    # Some released dumps misspell the communicative column.
    a = schema.resolve_columns(list(schema.CSV_COLUMNS))
    b = schema.resolve_columns(
        [c.replace("Communicative", "Comunicative") for c in schema.CSV_COLUMNS]
    )
    assert a["communicative"] == "attributes.Communicative"
    assert b["communicative"] == "attributes.Comunicative"
    assert a["obj_track_id"] == b["obj_track_id"] == "obj_track_id"


def test_all_canonical_keys_resolve_against_published_header():
    resolved = schema.resolve_columns(list(schema.CSV_COLUMNS))
    for key in schema.COLUMN_ALIASES:
        assert key in resolved, f"{key} did not resolve"


def test_action_group_lookup():
    atomic = schema.ACTION_GROUPS[0]
    assert schema.index_of(atomic, "walking") == atomic.classes.index("walking")
    assert schema.index_of(atomic, "WALKING") == atomic.classes.index("walking")
    assert schema.index_of(atomic, "") == -1
    assert schema.index_of(atomic, None) == -1
    assert schema.index_of(atomic, "not a real action") == -1


def test_split_sizes_match_release():
    assert schema.EXPECTED_SPLIT_SIZES == {"train": 400, "val": 200, "test": 100}
    assert sum(schema.EXPECTED_SPLIT_SIZES.values()) == 700


def test_horizons_match_paper_protocol():
    # 10 Hz annotations, 1 s observed, 2 s predicted.
    assert schema.OBS_LEN / schema.ANNOTATION_HZ == 1.0
    assert schema.PRED_LEN / schema.ANNOTATION_HZ == 2.0


def test_ltwh_conversion():
    cu, cv, lu, lv = ltwh_to_cxcywh(top=100.0, left=200.0, height=80.0, width=40.0)
    assert (cu, cv, lu, lv) == (220.0, 140.0, 40.0, 80.0)


def test_normalize_roundtrip():
    boxes = torch.tensor([[960.0, 600.0, 100.0, 200.0]])
    assert torch.allclose(denormalize_boxes(normalize_boxes(boxes)), boxes, atol=1e-4)
    assert torch.allclose(normalize_boxes(boxes)[0, :2], torch.tensor([0.5, 0.5]), atol=1e-6)


def test_paper_table_is_monotone_in_priors():
    # Guards the reference table against a typo: each added prior improves FDE.
    assert PAPER_RESULTS["EP+IP+AP"]["FDE"] < PAPER_RESULTS["EP+IP"]["FDE"]
    assert PAPER_RESULTS["EP+IP"]["FDE"] < PAPER_RESULTS["EP"]["FDE"]
    assert PAPER_RESULTS["EP"]["FDE"] < PAPER_RESULTS["vanilla"]["FDE"]
