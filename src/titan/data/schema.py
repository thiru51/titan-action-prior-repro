"""Schema of the released TITAN annotations.

Everything here mirrors the layout described on the Honda Research Institute
release page (usa.honda-ri.com/titan) and cross-checked against the TITAN
parser in vita-epfl/pedestrian-transition-dataset, which reads the real CSVs.
Nothing here has been run against the actual annotation tarballs yet -- see
the schema notes in HANDOFF.md before trusting the label vocabularies.
"""

from __future__ import annotations

from dataclasses import dataclass

# Column order of clip_xxx.csv inside titan_0_4.tar. The order matters: the
# vita-epfl parser addresses these positionally, and its positional drops line
# up exactly with the name list published by HRI, which is what makes this
# ordering trustworthy rather than guessed.
CSV_COLUMNS: tuple[str, ...] = (
    "frames",
    "label",
    "obj_track_id",
    "top",
    "left",
    "height",
    "width",
    "attributes.Trunk",
    "attributes.Motion Status",
    "attributes.Doors Open",
    "attributes.Communicative",
    "attributes.Complex Contextual",
    "attributes.Atomic Actions",
    "attributes.Simple Context",
    "attributes.Transporting",
    "attributes.Age",
)

# Real releases have drifted on these spellings ("Comunicative" is misspelled in
# some dumps), so the loader resolves columns by alias rather than by position.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "frames": ("frames", "frame_id", "frame"),
    "label": ("label", "class"),
    "obj_track_id": ("obj_track_id", "track_id", "obj_id"),
    "top": ("top",),
    "left": ("left",),
    "height": ("height",),
    "width": ("width",),
    "communicative": ("attributes.Communicative", "attributes.Comunicative"),
    "complex_contextual": ("attributes.Complex Contextual", "attributes.Complex"),
    "atomic": ("attributes.Atomic Actions", "attributes.Atomic"),
    "simple_contextual": ("attributes.Simple Context", "attributes.Simple"),
    "transporting": ("attributes.Transporting",),
    "age": ("attributes.Age",),
    "motion_status": ("attributes.Motion Status",),
}

PERSON_LABEL = "person"

# Reconstructed from the taxonomy table in the paper. Several groups carry a
# "none of the above" class, which is why the paper's per-group counts run one
# higher than the number of named actions.
NONE_LABEL = "none of the above"

ATOMIC_ACTIONS: tuple[str, ...] = (
    "standing",
    "walking",
    "running",
    "bending",
    "kneeling",
    "squatting",
    "sitting",
    "jumping",
    "laying down",
)

SIMPLE_CONTEXTUAL: tuple[str, ...] = (
    "crossing a street at pedestrian crossing",
    "jaywalking",
    "waiting to cross street",
    "motorcycling",
    "biking",
    "walking along the side of the road",
    "walking on the road",
    "cleaning an object",
    "closing",
    "opening",
    "entering a building",
    "exiting a building",
    NONE_LABEL,
)

COMPLEX_CONTEXTUAL: tuple[str, ...] = (
    "unloading",
    "loading",
    "getting in 4 wheel vehicle",
    "getting out of 4 wheel vehicle",
    "getting on 2 wheel vehicle",
    "getting off 2 wheel vehicle",
    NONE_LABEL,
)

COMMUNICATIVE: tuple[str, ...] = (
    "looking into phone",
    "talking on phone",
    "talking in group",
    NONE_LABEL,
)

TRANSPORTIVE: tuple[str, ...] = (
    "pushing",
    "carrying with both hands",
    "pulling",
    NONE_LABEL,
)

AGE_GROUPS: tuple[str, ...] = ("child", "adult", "senior")


@dataclass(frozen=True)
class ActionGroup:
    key: str
    column: str
    classes: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


# TITAN treats action recognition as several parallel multi-class heads rather
# than one flat softmax, and the paper's per-head uncertainty weighting only
# makes sense with that structure, so the grouping is kept explicit.
ACTION_GROUPS: tuple[ActionGroup, ...] = (
    ActionGroup("atomic", "atomic", ATOMIC_ACTIONS),
    ActionGroup("simple_contextual", "simple_contextual", SIMPLE_CONTEXTUAL),
    ActionGroup("complex_contextual", "complex_contextual", COMPLEX_CONTEXTUAL),
    ActionGroup("communicative", "communicative", COMMUNICATIVE),
    ActionGroup("transportive", "transportive", TRANSPORTIVE),
)

ACTION_GROUP_SIZES: tuple[int, ...] = tuple(g.num_classes for g in ACTION_GROUPS)
TOTAL_ACTION_CLASSES: int = sum(ACTION_GROUP_SIZES)

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

# Annotation sampling rate. Source video is 60 fps; the paper subsamples to
# 10 Hz, observes 1 s and forecasts 2 s.
ANNOTATION_HZ = 10
OBS_LEN = 10
PRED_LEN = 20

IMU_COLUMNS: tuple[str, ...] = (
    "image_ts",
    "image_path",
    "accel_ts",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_ts",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
)

SPLIT_FILES: dict[str, str] = {
    "train": "train_set.txt",
    "val": "val_set.txt",
    "test": "test_set.txt",
}

# Clip counts published with the release; the loader asserts against these so a
# partial download fails loudly instead of silently training on less data.
EXPECTED_SPLIT_SIZES: dict[str, int] = {"train": 400, "val": 200, "test": 100}


def resolve_columns(present: list[str]) -> dict[str, str]:
    """Map canonical keys onto whatever the CSV actually calls them."""
    lowered = {c.strip().lower(): c for c in present}
    resolved: dict[str, str] = {}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            hit = lowered.get(alias.strip().lower())
            if hit is not None:
                resolved[key] = hit
                break
    return resolved


def index_of(group: ActionGroup, value: object) -> int:
    """Class index for a raw cell value, or -1 when unlabelled."""
    if value is None:
        return -1
    text = str(value).strip().lower()
    if text in ("", "nan", "none", "na", "-1"):
        return -1
    for i, name in enumerate(group.classes):
        if text == name.lower():
            return i
    return -1
