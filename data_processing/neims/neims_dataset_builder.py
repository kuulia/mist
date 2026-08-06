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
    source_dirs: list        # list[Path], from repeated source_dir keys
    output_dir: Path
    dataset_name: str
    test_source_dirs: list   # list[Path], from repeated test_source_dir keys; ALL molecules → test
    n_val: int = 10240
    seed: int = 42
    balanced_val: bool = False  # split n_val evenly across regular sources, not by pool size
    log_level: str = "INFO"

    def __post_init__(self):
        self.source_dirs = [Path(d) for d in self.source_dirs]
        self.test_source_dirs = [Path(d) for d in self.test_source_dirs]
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
# Balanced validation sampling
# ---------------------------------------------------------------------------

def _balanced_val_split(
    dedup_labels: pd.DataFrame,
    non_test_names: set,
    n_val: int,
    seed: int,
) -> tuple:
    """
    Split non_test_names into (val_names, train_names), drawing val as evenly
    as possible across each regular source (identified by the "_source"
    column) rather than proportionally to each source's pool size.

    If a source's non-test pool is smaller than its equal share, the
    shortfall is redistributed across the remaining sources that still have
    spare capacity.
    """
    pool = dedup_labels[dedup_labels["spec"].isin(non_test_names)]
    pool_by_source = {
        src: grp["spec"].to_numpy()
        for src, grp in pool.groupby("_source")
    }
    sources = sorted(pool_by_source)
    n_sources = len(sources)

    rng = np.random.default_rng(seed)
    for src in sources:
        rng.shuffle(pool_by_source[src])

    base = n_val // n_sources
    remainder = n_val - base * n_sources
    target = {src: base for src in sources}
    for src in sources[:remainder]:
        target[src] += 1

    taken = {src: min(target[src], len(pool_by_source[src])) for src in sources}
    shortfall = sum(target[src] - taken[src] for src in sources)

    while shortfall > 0:
        capacity_sources = [s for s in sources if taken[s] < len(pool_by_source[s])]
        if not capacity_sources:
            break
        share = max(1, shortfall // len(capacity_sources))
        progress = False
        for s in capacity_sources:
            if shortfall <= 0:
                break
            extra = min(share, len(pool_by_source[s]) - taken[s], shortfall)
            if extra > 0:
                taken[s] += extra
                shortfall -= extra
                progress = True
        if not progress:
            break

    val_names: set = set()
    for src in sources:
        val_names.update(pool_by_source[src][:taken[src]].tolist())
        log.info(
            "Balanced val: %s contributes %d / %d available.",
            src, taken[src], len(pool_by_source[src]),
        )

    train_names = non_test_names - val_names
    return val_names, train_names


# ---------------------------------------------------------------------------
# Combination logic
# ---------------------------------------------------------------------------

def build_combined(cfg: Config) -> None:
    # ------------------------------------------------------------------
    # 1. Load all sources
    # ------------------------------------------------------------------
    regular_sources = [discover_source(d) for d in cfg.source_dirs]
    test_sources    = [discover_source(d) for d in cfg.test_source_dirs]
    all_sources = regular_sources + test_sources

    if test_sources:
        log.info(
            "Test-forced source(s): %s — ALL molecules from these will be locked to test.",
            [str(s.path) for s in test_sources],
        )

    total_input = sum(len(s.df) for s in all_sources)
    log.info(
        "Loaded %d source(s) (%d regular + %d test-forced) — %d spectra total.",
        len(all_sources), len(regular_sources), len(test_sources), total_input,
    )

    # ------------------------------------------------------------------
    # 2. Build InChIKey sets
    #
    # inchikey_from_test_sources: ALL IKs from test-forced sources.
    #   Used to (a) strip overlapping molecules from regular sources and
    #   (b) assign test labels in the combined split.
    #
    # Regular sources' own test-split IKs are NOT used here: all molecules
    # from regular sources (regardless of their original split label) go
    # into the combined train/val pool so the split can be re-randomised
    # from scratch.  Cross-source regular dedup (first-source-wins) prevents
    # the same molecule appearing in both train and val.
    # ------------------------------------------------------------------
    inchikey_from_test_sources: set = set()
    for s in test_sources:
        inchikey_from_test_sources.update(set(s.labels["inchikey"]) - {""})

    # inchikey_test_locked drives the final split assignment (only test sources).
    inchikey_test_locked: set = set(inchikey_from_test_sources)

    log.info(
        "Test-forced InChIKeys: %d (all molecules from test_source_dir entries).",
        len(inchikey_test_locked),
    )

    # ------------------------------------------------------------------
    # 4. Strip test-source molecules from regular sources
    #
    # Molecules whose InChIKey appears in any test-forced source must not
    # appear in the train/val pool.  Remove them so that the test-source
    # copy (not the regular-source copy) ends up in the dataset.
    # ------------------------------------------------------------------
    filtered_regular: list = []
    for s in regular_sources:
        is_test_locked = s.labels["inchikey"].isin(inchikey_from_test_sources)
        n_excluded = int(is_test_locked.sum())
        if n_excluded:
            log.info(
                "Stripped %d test-source-overlapping spectra from regular source %s.",
                n_excluded, s.path.name,
            )
        labels = s.labels[~is_test_locked].reset_index(drop=True)
        labels = labels.assign(_source=str(s.path))  # tag for balanced val sampling
        kept = set(labels["spec"])
        filtered_regular.append(SourceSet(
            path=s.path,
            df=s.df[s.df["name"].isin(kept)].reset_index(drop=True),
            labels=labels,
            split=s.split[s.split["name"].isin(kept)].reset_index(drop=True),
            subforms={k: v for k, v in s.subforms.items() if k in kept},
        ))

    tagged_test_sources = []
    for s in test_sources:
        labels = s.labels.assign(_source="__test__")
        tagged_test_sources.append(SourceSet(
            path=s.path, df=s.df, labels=labels, split=s.split, subforms=s.subforms,
        ))

    all_sources = filtered_regular + tagged_test_sources

    # ------------------------------------------------------------------
    # 5. Concatenate (after stripping) and deduplicate by InChIKey
    #
    # Duplicates can still occur within regular sources or within test
    # sources; first-source-wins resolves them.  There is no longer any
    # overlap between regular and test sources by construction.
    # ------------------------------------------------------------------
    all_df = pd.concat([s.df for s in all_sources], ignore_index=True)
    all_labels = pd.concat([s.labels for s in all_sources], ignore_index=True)
    all_subforms = {}
    for s in all_sources:
        overlap = set(s.subforms) & set(all_subforms)
        if overlap:
            log.warning(
                "%s: %d subform key collision(s) — later values overwrite.",
                s.path.name, len(overlap),
            )
        all_subforms.update(s.subforms)

    n_before = len(all_labels)
    has_key = all_labels["inchikey"].notna() & (all_labels["inchikey"] != "")
    dedup_with_key = all_labels[has_key].drop_duplicates(subset=["inchikey"], keep="first")
    dedup_without_key = all_labels[~has_key]
    dedup_labels = pd.concat([dedup_with_key, dedup_without_key], ignore_index=True)
    n_removed = n_before - len(dedup_labels)
    if n_removed:
        log.info(
            "Deduplication: removed %d duplicate InChIKey spectra. %d remain.",
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
    #    val   = n_val molecules from the remaining non-test pool
    #    train = everything else
    #
    # If cfg.balanced_val is set, val is split as evenly as possible across
    # each regular source (by source count, not pool size) instead of being
    # drawn proportionally from the pooled non-test set.
    # ------------------------------------------------------------------
    name_to_ik = dict(zip(dedup_labels["spec"], dedup_labels["inchikey"]))
    test_names = {
        name for name in kept_names
        if name_to_ik.get(name, "") in inchikey_test_locked
    }
    non_test_names = kept_names - test_names

    if cfg.balanced_val and len(regular_sources) > 1:
        val_names, train_names = _balanced_val_split(
            dedup_labels, non_test_names, cfg.n_val, cfg.seed,
        )
    else:
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

    # Drop the internal source-tracking column before writing labels.tsv
    dedup_labels = dedup_labels.drop(columns=["_source"])

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
# source_dir      — repeat for each train/val source; existing split files are
#                   respected (molecules already in "test" stay in test).
# test_source_dir — repeat for each source whose molecules should ALL go to
#                   test, ignoring their existing split assignments.

source_dir: data/neims/gecko_EIMS_spectra/mist_inputs/gecko
source_dir: data/neims/msg/mist_inputs

test_source_dir: data/neims/atmomaccs_new/mist_inputs

# Output directory for the combined dataset artifacts.
output_dir: data/neims/combined/mist_inputs

# Name used in output filenames:
#   df_neims_NAME_3_9_22.pkl
#   NAME_subforms_3_9_22.pkl
dataset_name: combined

# Number of spectra to draw for the validation set from the non-test pool.
# Capped at 20 %% of the non-test pool to avoid edge cases on small datasets
# (cap is not applied when balanced_val is true).
n_val: 10240

# When true and there are multiple source_dir entries, n_val is split as
# evenly as possible across each regular source instead of being drawn
# proportionally to each source's pool size.
balanced_val: false

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

    raw_test_dirs = data.get("test_source_dir", [])
    if not isinstance(raw_test_dirs, list):
        raw_test_dirs = [raw_test_dirs]
    test_source_dirs = [Path(d) for d in raw_test_dirs if d is not None]

    if not source_dirs and not test_source_dirs:
        raise ValueError(
            f"Config {path} has no 'source_dir' or 'test_source_dir' entries."
        )

    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in valid_levels:
        raise ValueError(f"log_level must be one of {valid_levels}, got {log_level!r}")

    return Config(
        source_dirs=source_dirs,
        test_source_dirs=test_source_dirs,
        output_dir=Path(data["output_dir"]),
        dataset_name=str(data["dataset_name"]),
        n_val=int(data.get("n_val", 10240)),
        seed=int(data.get("seed", 42)),
        balanced_val=bool(data.get("balanced_val", False)),
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
