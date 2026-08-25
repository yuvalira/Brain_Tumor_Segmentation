import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROJECT_ROOT, RANDOM_SEED, SLICE_NUM, TOTAL_VOLUMES
from utilities.utils import DATASET_DIR


SPLIT_PATH = Path(PROJECT_ROOT) / "saved_parameters" / "patient_split.json"
SPLIT_VERSION = "central_slice_size_and_order_stratified_v1"


def _central_slice_tumor_pixels(volume):
    path = Path(DATASET_DIR) / f"volume_{volume}_slice_{SLICE_NUM}.h5"
    with h5py.File(path, "r") as file:
        mask = file["mask"][:]
    tumor = np.any(mask > 0, axis=-1) if mask.ndim == 3 else mask > 0
    return int(tumor.sum())


def _size_groups(tumor_pixels):
    tumor_pixels = np.asarray(tumor_pixels)
    positive = tumor_pixels[tumor_pixels > 0]
    cutoffs = np.unique(np.quantile(positive, [0.25, 0.50, 0.75]))
    groups = np.full(len(tumor_pixels), "no tumor", dtype=object)
    labels = ["small", "medium", "large", "very large"]
    groups[tumor_pixels > 0] = [
        labels[min(np.digitize(value, cutoffs), len(labels) - 1)]
        for value in tumor_pixels[tumor_pixels > 0]
    ]
    return groups, cutoffs.tolist()


def _split_id(train, validation, test):
    payload = json.dumps({
        "train": list(map(int, train)),
        "validation": list(map(int, validation)),
        "test": list(map(int, test)),
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _validate_split(split, train_size, validation_size, test_size):
    train = list(map(int, split["train"]))
    validation = list(map(int, split["validation"]))
    test = list(map(int, split["test"]))
    if [len(train), len(validation), len(test)] != [
        train_size, validation_size, test_size
    ]:
        raise ValueError("Saved patient split has incorrect subset sizes.")
    combined = train + validation + test
    if len(set(combined)) != TOTAL_VOLUMES or set(combined) != set(
        range(1, TOTAL_VOLUMES + 1)
    ):
        raise ValueError("Patient split must contain every patient exactly once.")
    if split.get("split_id") != _split_id(train, validation, test):
        raise ValueError("Saved patient split ID does not match its patient lists.")


def create_or_load_patient_split(
    path=SPLIT_PATH,
    train_size=250,
    validation_size=50,
    test_size=69,
    seed=RANDOM_SEED,
    force_recreate=False,
):
    """Create one reproducible patient split for the central-slice experiment.

    Stratification combines central-slice tumor-size groups with patient-index
    quartiles. The latter prevents contiguous source/order blocks from being
    isolated in one subset when clinic identifiers are unavailable in the HDF5
    files. The exact split is saved and reused by every model.
    """
    path = Path(path)
    if path.exists() and not force_recreate:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("split_version") == SPLIT_VERSION:
            _validate_split(saved, train_size, validation_size, test_size)
            return saved

    if train_size + validation_size + test_size != TOTAL_VOLUMES:
        raise ValueError("Split sizes must sum to TOTAL_VOLUMES.")

    volumes = np.arange(1, TOTAL_VOLUMES + 1)
    tumor_pixels = np.asarray([
        _central_slice_tumor_pixels(volume) for volume in volumes
    ])
    size_groups, size_cutoffs = _size_groups(tumor_pixels)
    order_quartiles = np.minimum((volumes - 1) * 4 // TOTAL_VOLUMES, 3)
    strata = np.asarray([
        f"{size_group}|order-{order_quartile}"
        for size_group, order_quartile in zip(size_groups, order_quartiles)
    ])

    train, remaining = train_test_split(
        volumes,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    remaining_strata = strata[remaining - 1]
    validation, test = train_test_split(
        remaining,
        train_size=validation_size,
        test_size=test_size,
        random_state=seed + 1,
        shuffle=True,
        stratify=remaining_strata,
    )

    split = {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "slice_num": SLICE_NUM,
        "stratification": "central-slice tumor-size group + patient-index quartile",
        "clinic_note": (
            "The HDF5 files do not contain clinic identifiers. Patient-index "
            "quartiles are balanced to prevent ordered source blocks from being "
            "isolated, but this is not a substitute for true clinic metadata."
        ),
        "tumor_size_cutoffs_pixels": size_cutoffs,
        "split_id": _split_id(train, validation, test),
        "train": train.astype(int).tolist(),
        "validation": validation.astype(int).tolist(),
        "test": test.astype(int).tolist(),
        "patients": [
            {
                "volume": int(volume),
                "tumor_pixels": int(pixels),
                "tumor_size_group": str(size_group),
                "order_quartile": int(order_quartile),
            }
            for volume, pixels, size_group, order_quartile in zip(
                volumes, tumor_pixels, size_groups, order_quartiles
            )
        ],
    }
    _validate_split(split, train_size, validation_size, test_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def split_summary(split):
    patient_table = pd.DataFrame(split["patients"])
    membership = {
        int(volume): subset
        for subset in ["train", "validation", "test"]
        for volume in split[subset]
    }
    patient_table["split"] = patient_table["volume"].map(membership)
    return patient_table


def model_matches_training_split(model_path, training_volumes):
    model_path = Path(model_path)
    if not model_path.exists():
        return False
    with np.load(model_path) as parameters:
        return (
            "training_volumes" in parameters
            and np.array_equal(
                parameters["training_volumes"], np.asarray(training_volumes)
            )
        )
