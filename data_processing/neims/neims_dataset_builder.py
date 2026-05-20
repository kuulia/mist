"""
neims_dataset_builder.py — Combine multiple preprocessed NEIMS datasets.

Takes the output directories of neims_preprocessing.py (one per source dataset),
concatenates and deduplicates by InChIKey, then writes the four MIST artifacts
for the combined dataset.

Split strategy: "preserve test, re-randomise val/train"
  - Test-locking is done at the InChIKey level BEFORE deduplication: if a molecule
    appears in ANY source's test split, it is locked to test in the combined output.
    This prevents test contamination even when the same molecule appears under a
    different spectrum name in another source's train split.
  - After deduplication (first-source-wins), the remaining non-test pool is randomly
    split into val (n_val) and train (rest).

File auto-discovery
-------------------
Each source_dir is expected to be the output_dir of neims_preprocessing.py and must
contain exactly one of each:
  df_neims_*_3_9_22.pkl
  labels.tsv
  split_*.tsv
  *_subforms_3_9_22.pkl

Usage
-----
  python neims_dataset_builder.py config.yaml
  python neims_dataset_builder.py --template > config.yaml
"""

import argparse
import dataclasses
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    source_dirs: list   # list[Path], populated from repeated source_dir keys
    output_dir: Path
    dataset_name: str
    n_val: int = 10240
    seed: int = 42
    log_level: str = "INFO"

    def __post_init__(self):
        self.source_dirs = [Path(d) for d in self.source_dirs]
        self.output_dir = Path(self.output_dir)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SourceSet:
    path: Path
    df: pd.DataFrame      # columns: SMILES, spec, name
    labels: pd.DataFrame  # columns: dataset, spec, ionization, formula, smiles, inchikey, instrument
    split: pd.DataFrame   # columns: name, split
    subforms: dict        # {name: json_str}


def _require_one(files: list, pattern: str, source_dir: Path) -> Path:
    if not files:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {source_dir}. "
            "Make sure this directory was produced by neims_preprocessing.py."
        )
    if len(files) > 1:
        names = [f.name for f in files]
        raise RuntimeError(
            f"Multiple files matching '{pattern}' in {source_dir}: {names}. "
            "Expected exactly one — remove or rename the extra file."
        )
    return files[0]


def discover_source(source_dir: Path) -> SourceSet:
    """Load all four MIST artifacts from a neims_preprocessing.py output directory."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    df_file = _require_one(
        sorted(source_dir.glob("df_neims_*_3_9_22.pkl")),
        "df_neims_*_3_9_22.pkl", source_dir,
    )
    labels_file = source_dir / "labels.tsv"
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.tsv not found in {source_dir}")
    split_file = _require_one(
        sorted(source_dir.glob("split_*.tsv")),
        "split_*.tsv", source_dir,
    )
    subforms_file = _require_one(
        sorted(source_dir.glob("*_subforms_3_9_22.pkl")),
        "*_subforms_3_9_22.pkl", source_dir,
    )

    log.info(
        "Loading source %s  (df=%s, split=%s, subforms=%s)",
        source_dir, df_file.name, split_file.name, subforms_file.name,
    )

    df = pd.read_pickle(df_file)
    labels = pd.read_csv(labels_file, sep="\t")
    split = pd.read_csv(split_file, sep="\t")
    with open(subforms_file, "rb") as fh:
        subforms = pickle.load(fh)

    log.info("  → %d spectra", len(df))
    return SourceSet(
        path=source_dir,
        df=df,
        labels=labels,
        split=split,
        subforms=subforms,
    )


# ---------------------------------------------------------------------------
# Combination logic
# ---------------------------------------------------------------------------

def build_combined(cfg: Config) -> None:
    # ------------------------------------------------------------------
    # 1. Load all sources
    # ------------------------------------------------------------------
    sources = [discover_source(d) for d in cfg.source_dirs]
    total_input = sum(len(s.df) for s in sources)
    log.info(
        "Loaded %d source(s) — %d spectra total before deduplication.",
        len(sources), total_input,
    )

    # ------------------------------------------------------------------
    # 2. Concatenate DataFrames, labels, subforms
    # ------------------------------------------------------------------
    all_df = pd.concat([s.df for s in sources], ignore_index=True)
    all_labels = pd.concat([s.labels for s in sources], ignore_index=True)

    all_subforms: dict = {}
    for s in sources:
        overlap = set(s.subforms) & set(all_subforms)
        if overlap:
            log.warning(
                "%s: %d subform key(s) already seen from a prior source — "
                "later values will overwrite. Check that source datasets have "
                "unique id_prefix values.",
                s.path.name, len(overlap),
            )
        all_subforms.update(s.subforms)

    # ------------------------------------------------------------------
    # 3. Build test-locked InChIKey set BEFORE deduplication
    #
    # A molecule is test-locked if its InChIKey appears in ANY source's
    # test split.  This prevents the same molecule appearing in test via
    # one source and train via another from causing contamination.
    # ------------------------------------------------------------------
    inchikey_test_locked: set = set()
    for s in sources:
        test_spec_names = set(s.split.loc[s.split["split"] == "test", "name"])
        locked = set(
            s.labels.loc[s.labels["spec"].isin(test_spec_names), "inchikey"]
        ) - {""}
        inchikey_test_locked.update(locked)

    log.info(
        "Test-locked InChIKeys: %d (union across all sources before dedup).",
        len(inchikey_test_locked),
    )

    # ------------------------------------------------------------------
    # 4. Deduplicate by InChIKey — first-source-wins
    #
    # Molecules with an empty InChIKey (RDKit failure) are never merged;
    # each is kept as-is regardless of whether others share the same entry.
    # ------------------------------------------------------------------
    n_before = len(all_labels)
    has_key = all_labels["inchikey"].notna() & (all_labels["inchikey"] != "")
    dedup_with_key = all_labels[has_key].drop_duplicates(subset=["inchikey"], keep="first")
    dedup_without_key = all_labels[~has_key]
    dedup_labels = pd.concat([dedup_with_key, dedup_without_key], ignore_index=True)
    n_removed = n_before - len(dedup_labels)
    if n_removed:
        log.info(
            "Deduplication: removed %d spectrum/spectra with duplicate InChIKey. "
            "%d remain.",
            n_removed, len(dedup_labels),
        )

    kept_names: set = set(dedup_labels["spec"])

    # Filter df and subforms to kept names
    dedup_df = all_df[all_df["name"].isin(kept_names)].reset_index(drop=True)
    dedup_subforms = {k: v for k, v in all_subforms.items() if k in kept_names}

    missing_sub = kept_names - set(dedup_subforms)
    if missing_sub:
        log.warning(
            "%d kept spectra have no subforms entry. Examples: %s",
            len(missing_sub), sorted(missing_sub)[:5],
        )

    # ------------------------------------------------------------------
    # 5. Build split
    #    test  = any molecule whose InChIKey was locked from a source test split
    #    val   = n_val random molecules from the remaining non-test pool
    #    train = everything else
    # ------------------------------------------------------------------
    name_to_ik = dict(zip(dedup_labels["spec"], dedup_labels["inchikey"]))
    test_names = {
        name for name in kept_names
        if name_to_ik.get(name, "") in inchikey_test_locked
    }
    non_test_names = kept_names - test_names

    non_test_arr = np.array(sorted(non_test_names))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(non_test_arr)

    n_val = min(cfg.n_val, len(non_test_arr) // 5)
    val_names = set(non_test_arr[:n_val].tolist())
    train_names = non_test_names - val_names

    split_df = pd.DataFrame(
        [{"name": n, "split": "train"} for n in sorted(train_names)]
        + [{"name": n, "split": "val"}   for n in sorted(val_names)]
        + [{"name": n, "split": "test"}  for n in sorted(test_names)]
    )

    log.info(
        "Combined split: %d train | %d val | %d test  (total %d spectra)",
        len(train_names), len(val_names), len(test_names), len(dedup_df),
    )

    # ------------------------------------------------------------------
    # 6. Write the four MIST artifacts
    # ------------------------------------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df_out = cfg.output_dir / f"df_neims_{cfg.dataset_name}_3_9_22.pkl"
    dedup_df.to_pickle(df_out)
    log.info("Wrote %s", df_out.name)

    labels_out = cfg.output_dir / "labels.tsv"
    dedup_labels.to_csv(labels_out, sep="\t", index=False)
    log.info("Wrote labels.tsv (%d rows)", len(dedup_labels))

    split_out = cfg.output_dir / "split_random.tsv"
    split_df.to_csv(split_out, sep="\t", index=False)
    log.info("Wrote split_random.tsv")

    subforms_out = cfg.output_dir / f"{cfg.dataset_name}_subforms_3_9_22.pkl"
    with open(subforms_out, "wb") as fh:
        pickle.dump(dedup_subforms, fh)
    log.info("Wrote %s (%d entries)", subforms_out.name, len(dedup_subforms))


# ---------------------------------------------------------------------------
# YAML config loader  (flat key: value; repeated keys accumulate into a list)
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# neims_dataset_builder configuration
# Run with:  python neims_dataset_builder.py config.yaml
#
# List all source directories by repeating the source_dir key.
# Each must be an output directory of neims_preprocessing.py.

source_dir: data/neims/gecko_EIMS_spectra/mist_inputs
source_dir: data/neims/msg/mist_inputs

# Output directory for the combined dataset artifacts.
output_dir: data/neims/combined/mist_inputs

# Name used in output filenames:
#   df_neims_NAME_3_9_22.pkl
#   NAME_subforms_3_9_22.pkl
dataset_name: combined

# Number of spectra to draw for the validation set from the non-test pool.
# Capped at 20 %% of the non-test pool to avoid edge cases on small datasets.
n_val: 10240

# Random seed for val set selection.
seed: 42

# Logging verbosity: DEBUG, INFO, WARNING, or ERROR.
log_level: INFO
"""


def _parse_scalar(raw: str):
    raw = raw.split(" #")[0].split("\t#")[0]
    s = raw.strip()
    if s in ("null", "~", ""):
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def load_yaml_config(path: Path) -> Config:
    """
    Parse a flat YAML config file.

    Repeated keys are accumulated into a list (used for source_dir).
    """
    data: dict = {}
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"{path}:{lineno}: expected 'key: value', got: {line!r}"
                )
            key, _, value = line.partition(":")
            key = key.strip()
            val = _parse_scalar(value)
            if key in data:
                if not isinstance(data[key], list):
                    data[key] = [data[key]]
                data[key].append(val)
            else:
                data[key] = val

    for req in ("output_dir", "dataset_name"):
        if not data.get(req):
            raise ValueError(f"Config {path} is missing required key: '{req}'")

    raw_dirs = data.get("source_dir", [])
    if not isinstance(raw_dirs, list):
        raw_dirs = [raw_dirs]
    source_dirs = [Path(d) for d in raw_dirs if d is not None]
    if not source_dirs:
        raise ValueError(
            f"Config {path} has no 'source_dir' entries. "
            "Add one or more lines: source_dir: <path>"
        )

    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in valid_levels:
        raise ValueError(f"log_level must be one of {valid_levels}, got {log_level!r}")

    return Config(
        source_dirs=source_dirs,
        output_dir=Path(data["output_dir"]),
        dataset_name=str(data["dataset_name"]),
        n_val=int(data.get("n_val", 10240)),
        seed=int(data.get("seed", 42)),
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "config",
        nargs="?",
        metavar="CONFIG",
        help="Path to a YAML config file (see --template for format).",
    )
    p.add_argument(
        "--template",
        action="store_true",
        help="Print a template config file to stdout and exit.",
    )
    return p


def main(argv=None) -> None:
    p = _build_parser()
    args = p.parse_args(argv)

    if args.template:
        sys.stdout.write(_TEMPLATE)
        sys.exit(0)

    if not args.config:
        p.error("Provide a config file, or use --template to generate one.")

    cfg = load_yaml_config(Path(args.config))

    logging.basicConfig(
        level=getattr(logging, cfg.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    build_combined(cfg)


if __name__ == "__main__":
    main()
