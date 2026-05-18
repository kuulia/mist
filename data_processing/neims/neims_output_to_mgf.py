"""
neims_output_to_mgf.py  –  Convert NEIMS output folders to a single MGF file.

NEIMS produces one folder per molecule (000000/, 000001/, …), each containing
an annotated.sdf with a predicted EI-MS spectrum.  A separate CSV holds the
corresponding SMILES.  This script aligns them by row index and writes every
valid molecule-spectrum pair to one MGF file.

Alignment guarantee
-------------------
Row *i* (0-based) of the CSV maps to folder ``f"{i:06d}"``.  The script
verifies alignment by comparing the heavy-atom count read from the SDF mol
block against the atom count computed from the SMILES via RDKit.  Mismatches
are logged and skipped; they indicate a corrupt or misaligned dataset.

Dependencies: Python ≥ 3.8, pandas, numpy, rdkit

Usage
-----
  # Run with a YAML config file:
  python neims_output_to_mgf.py config.yaml

  # Print a template config to stdout, then edit it:
  python neims_output_to_mgf.py --template > config.yaml
"""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass  (single source of truth for all parameters)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    spectra_dir: Path
    smiles_csv: Path
    output: Path
    smiles_col: Optional[str] = None
    id_prefix: Optional[str] = None
    min_peaks: int = 5
    log_level: str = "INFO"

    def __post_init__(self):
        self.spectra_dir = Path(self.spectra_dir)
        self.smiles_csv = Path(self.smiles_csv)
        self.output = Path(self.output)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_smiles(csv_path: Path, smiles_col: Optional[str] = None) -> pd.Series:
    """
    Load a SMILES Series from csv_path, indexed 0, 1, 2, …

    Column selection (tried in order when smiles_col is None):
      'SMILES', 'smiles', 'Modified_SMILES', 'Original_SMILES'

    Unnamed index columns (Unnamed: 0, etc.) are dropped before indexing so
    that the positional row number matches the folder name regardless of
    whether the CSV was saved with or without an explicit index.
    """
    df = pd.read_csv(csv_path)

    # Drop any auto-saved pandas index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        log.debug("Dropping index column(s): %s", unnamed)
        df = df.drop(columns=unnamed)

    if smiles_col is not None:
        if smiles_col not in df.columns:
            raise ValueError(
                f"Column '{smiles_col}' not found. Available: {list(df.columns)}"
            )
        col = smiles_col
    else:
        candidates = ["SMILES", "smiles", "Modified_SMILES", "Original_SMILES"]
        found = [c for c in candidates if c in df.columns]
        if not found:
            raise ValueError(
                f"Cannot find a SMILES column in {list(df.columns)}. "
                f"Tried: {candidates}. Use --smiles-col to specify."
            )
        col = found[0]
        if len(found) > 1:
            log.warning(
                "Multiple SMILES-like columns found %s, using '%s'. "
                "Use --smiles-col to override.",
                found,
                col,
            )

    log.info("Using SMILES column '%s' from %s", col, csv_path.name)
    return df[col].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SDF parsing
# ---------------------------------------------------------------------------

def _mol_block_heavy_atom_count(lines: list[str]) -> Optional[int]:
    """
    Extract the heavy-atom count from a V2000 mol block counts line.

    The counts line is the fourth line of a mol block:
      aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    First 3 chars = atom count, next 3 = bond count.
    Returns None if the line cannot be parsed.
    """
    # Find the first line that looks like a V2000 counts line
    for line in lines:
        stripped = line.rstrip("\n")
        if "V2000" in stripped or "V3000" in stripped:
            try:
                return int(stripped[:3].strip())
            except ValueError:
                return None
    return None


def _parse_spectrum(lines: list[str]) -> Optional[np.ndarray]:
    """
    Parse the PREDICTED SPECTRUM block from an annotated.sdf.

    Returns an (N, 2) uint16 array of [mz, intensity] pairs,
    or None if no spectrum is found or it has fewer than min_peaks entries.
    The caller supplies min_peaks via the enclosing scope.
    """
    in_spec = False
    mz_vals: list[int] = []
    int_vals: list[int] = []

    for line in lines:
        lo = line.lower().strip()
        if "predicted spectrum" in lo:
            in_spec = True
            continue
        if in_spec:
            if lo in ("$$$$", ""):
                break
            parts = lo.split()
            if len(parts) != 2:
                continue
            try:
                mz_vals.append(int(parts[0]))
                int_vals.append(int(parts[1]))
            except ValueError:
                continue

    if not mz_vals:
        return None
    return np.array([mz_vals, int_vals], dtype=np.uint16).T


def parse_sdf(sdf_path: Path, min_peaks: int = 5) -> tuple:
    """
    Parse an annotated.sdf file.

    Returns (spectrum, heavy_atom_count).
    spectrum is an (N, 2) uint16 array or None on failure.
    heavy_atom_count is the atom count from the mol block, or None.
    """
    if not sdf_path.exists():
        return None, None
    try:
        with open(sdf_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        log.warning("Cannot read %s: %s", sdf_path, exc)
        return None, None

    heavy_count = _mol_block_heavy_atom_count(lines)
    spectrum = _parse_spectrum(lines)

    if spectrum is None:
        return None, heavy_count
    if len(spectrum) < min_peaks:
        return None, heavy_count

    return spectrum, heavy_count


# ---------------------------------------------------------------------------
# SMILES validation
# ---------------------------------------------------------------------------

def validate_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Return an RDKit Mol or None if the SMILES cannot be parsed."""
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def check_alignment(
    mol: Chem.Mol,
    sdf_heavy_count: Optional[int],
    folder: str,
) -> bool:
    """
    Compare the heavy-atom count from the SDF mol block with the count
    computed from the SMILES.  Returns True if they agree (or if the SDF
    count is unavailable), False on a mismatch.
    """
    if sdf_heavy_count is None:
        return True  # cannot verify; assume ok
    smiles_heavy = mol.GetNumHeavyAtoms()
    if smiles_heavy != sdf_heavy_count:
        log.warning(
            "Alignment mismatch in folder %s: "
            "SMILES has %d heavy atoms, SDF mol block has %d — skipping.",
            folder,
            smiles_heavy,
            sdf_heavy_count,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# MGF writer
# ---------------------------------------------------------------------------

def write_mgf(
    records: list[dict],
    output_path: Path,
) -> None:
    """
    Write a list of records to an MGF file.

    Each record must have: feature_id, smiles, mol, spectrum (N,2 array).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for rec in records:
            pmass = ExactMolWt(rec["mol"])
            fh.write("BEGIN IONS\n")
            fh.write(f"FEATURE_ID={rec['feature_id']}\n")
            fh.write(f"SMILES={rec['smiles']}\n")
            fh.write(f"PEPMASS={pmass:.6f}\n")
            fh.write("CHARGE=1+\n")
            fh.write("MSLEVEL=1\n")
            fh.write("IONIZATION_MODE=EI\n")
            fh.write(f"TITLE={rec['feature_id']}\n")
            spec = rec["spectrum"]
            for mz, intensity in spec:
                fh.write(f"{int(mz)} {int(intensity)}\n")
            fh.write("END IONS\n\n")


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert(cfg: Config) -> None:
    """
    Align SMILES from cfg.smiles_csv with spectra from cfg.spectra_dir and
    write an MGF file to cfg.output.
    """
    spectra_dir = cfg.spectra_dir
    smiles_csv = cfg.smiles_csv
    output_path = cfg.output
    smiles_col = cfg.smiles_col
    min_peaks = cfg.min_peaks
    id_prefix = cfg.id_prefix if cfg.id_prefix is not None else spectra_dir.name

    # ------------------------------------------------------------------
    # 1. Load SMILES
    # ------------------------------------------------------------------
    smiles_series = load_smiles(smiles_csv, smiles_col)
    n_rows = len(smiles_series)
    log.info("Loaded %d SMILES rows from %s", n_rows, smiles_csv.name)

    # ------------------------------------------------------------------
    # 2. Discover which numbered folders actually exist
    # ------------------------------------------------------------------
    existing_folders: set[str] = set()
    for entry in spectra_dir.iterdir():
        if entry.is_dir() and (entry / "annotated.sdf").exists():
            existing_folders.add(entry.name)

    n_folders = len(existing_folders)
    log.info("Found %d annotated.sdf folders in %s", n_folders, spectra_dir)

    if n_folders == 0:
        raise RuntimeError(
            f"No folders containing annotated.sdf found in {spectra_dir}. "
            "Check --spectra-dir points to the directory of numbered folders."
        )

    # Warn when counts differ substantially
    if abs(n_folders - n_rows) > max(1, 0.01 * n_rows):
        log.warning(
            "Folder count (%d) and CSV row count (%d) differ by more than 1%%. "
            "Check that --spectra-dir and --smiles-csv are matched.",
            n_folders,
            n_rows,
        )

    # ------------------------------------------------------------------
    # 3. Iterate over CSV rows, load matching SDF, validate, collect
    #
    # Both conditions (invalid SMILES, missing folder) are checked for every
    # row so they can be cross-referenced in the summary.  A missing folder
    # on a row whose SMILES is also invalid most likely means NEIMS itself
    # could not process that molecule and simply produced no output.
    # ------------------------------------------------------------------
    records: list[dict] = []
    invalid_smiles_idxs: set[int] = set()
    missing_folder_idxs: set[int] = set()
    n_bad_spectrum = 0
    n_misaligned = 0

    for idx, smiles in smiles_series.items():
        folder_name = f"{int(idx):06d}"

        mol = validate_smiles(smiles)
        folder_missing = folder_name not in existing_folders

        if mol is None:
            invalid_smiles_idxs.add(int(idx))
            if folder_missing:
                # NEIMS likely failed on this molecule — no folder produced
                log.debug(
                    "Row %d: invalid SMILES and no output folder "
                    "(NEIMS likely skipped this molecule): %s",
                    idx, smiles,
                )
            else:
                log.debug("Row %d: invalid SMILES '%s' — skipping.", idx, smiles)
            continue

        if folder_missing:
            missing_folder_idxs.add(int(idx))
            log.warning(
                "Row %d: folder %s not found but SMILES is valid — "
                "NEIMS may have failed or crashed on this molecule: %s",
                idx, folder_name, smiles,
            )
            continue

        sdf_path = spectra_dir / folder_name / "annotated.sdf"
        spectrum, sdf_heavy = parse_sdf(sdf_path, min_peaks=min_peaks)

        if spectrum is None:
            log.debug("Row %d: no valid spectrum in %s — skipping.", idx, folder_name)
            n_bad_spectrum += 1
            continue

        if not check_alignment(mol, sdf_heavy, folder_name):
            n_misaligned += 1
            continue

        records.append(
            {
                "feature_id": f"{id_prefix}_{idx}",
                "smiles": smiles,
                "mol": mol,
                "spectrum": spectrum,
            }
        )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    # Cross-reference invalid SMILES with missing folders to distinguish
    # "NEIMS skipped it" from "NEIMS ran but something else went wrong".
    also_missing = invalid_smiles_idxs & {
        int(f) for f in set(f"{i:06d}" for i in range(n_rows)) - existing_folders
    }
    only_bad_smiles = invalid_smiles_idxs - also_missing

    log.info("Kept    : %d spectra", len(records))
    if invalid_smiles_idxs:
        log.warning(
            "Skipped : %d rows with invalid SMILES "
            "(%d also have no output folder — NEIMS likely skipped these; "
            "%d have a folder despite bad SMILES)",
            len(invalid_smiles_idxs),
            len(also_missing),
            len(only_bad_smiles),
        )
    if missing_folder_idxs:
        log.warning(
            "Skipped : %d rows whose folder is missing but SMILES is valid "
            "(unexpected — NEIMS may have crashed on these): rows %s",
            len(missing_folder_idxs),
            sorted(missing_folder_idxs),
        )
    if n_bad_spectrum:
        log.warning("Skipped : %d folders with missing/short spectrum", n_bad_spectrum)
    if n_misaligned:
        log.warning("Skipped : %d rows with atom-count mismatch (alignment check)", n_misaligned)

    if not records:
        raise RuntimeError("No valid records found. Nothing written.")

    # ------------------------------------------------------------------
    # 5. Row-accounting invariant
    # Every input row must land in exactly one bucket.  A violated assert
    # means our loop logic has a bug (silent drop or double-count).
    # ------------------------------------------------------------------
    n_accounted = (
        len(records)
        + len(invalid_smiles_idxs)
        + len(missing_folder_idxs)
        + n_bad_spectrum
        + n_misaligned
    )
    if n_accounted != n_rows:
        raise RuntimeError(
            f"Row accounting mismatch: {n_accounted} accounted for but "
            f"{n_rows} rows were loaded. This is a bug — please report it."
        )
    log.debug("Row accounting OK: all %d rows accounted for.", n_rows)

    # ------------------------------------------------------------------
    # 6. Write MGF
    # ------------------------------------------------------------------
    write_mgf(records, output_path)

    # Verify the written file contains exactly the expected number of spectra.
    written = output_path.read_text().count("BEGIN IONS")
    if written != len(records):
        raise RuntimeError(
            f"MGF write verification failed: expected {len(records)} spectra "
            f"in {output_path} but counted {written} 'BEGIN IONS' blocks."
        )
    log.info("Wrote %d spectra to %s (verified)", len(records), output_path)


# ---------------------------------------------------------------------------
# YAML config loader  (no external dependency – parses flat key: value only)
# ---------------------------------------------------------------------------

# Template printed by --template.  The file the user edits and passes to the script.
_TEMPLATE = """\
# neims_output_to_mgf configuration
# Run with:  python neims_output_to_mgf.py config.yaml

# --- Required ---

# Directory containing NEIMS numbered output folders (000000/, 000001/, …).
# Each sub-folder must contain an annotated.sdf file.
spectra_dir: data/neims/my_dataset/NEIMS/original

# CSV file whose row i (0-based) corresponds to folder i.
# An auto-saved pandas index column (Unnamed: 0) is dropped automatically.
smiles_csv:  data/neims/my_dataset/compounds/original/dataset.csv

# Destination MGF file (parent directory is created if it does not exist).
output:      data/neims/my_dataset/spectra.mgf

# --- Optional ---

# SMILES column name in the CSV.
# Leave as null to auto-detect from: SMILES, smiles, Modified_SMILES, Original_SMILES.
smiles_col: null

# Prefix used for FEATURE_ID in the MGF (e.g. "gecko" → FEATURE_ID=gecko_0).
# Leave as null to use the name of the spectra_dir folder.
id_prefix: null

# Minimum number of peaks a spectrum must have to be kept.
min_peaks: 5

# Logging verbosity: DEBUG, INFO, WARNING, or ERROR.
log_level: INFO
"""


def _parse_scalar(raw: str):
    """Convert a raw YAML scalar string to an appropriate Python type."""
    # Strip inline comments: YAML requires a space before '#'
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
    # Strip optional surrounding quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def load_yaml_config(path: Path) -> Config:
    """
    Parse a flat YAML config file into a Config object.

    Only top-level key: value pairs are supported (no nested mappings or
    lists).  Lines beginning with '#' and blank lines are ignored.
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
            data[key] = _parse_scalar(value)

    required = ("spectra_dir", "smiles_csv", "output")
    missing = [k for k in required if k not in data or data[k] is None]
    if missing:
        raise ValueError(
            f"Config file {path} is missing required keys: {missing}"
        )

    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR")
    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in valid_levels:
        raise ValueError(
            f"log_level must be one of {valid_levels}, got {log_level!r}"
        )

    min_peaks = data.get("min_peaks", 5)
    if not isinstance(min_peaks, int) or min_peaks < 1:
        raise ValueError(f"min_peaks must be a positive integer, got {min_peaks!r}")

    return Config(
        spectra_dir=Path(data["spectra_dir"]),
        smiles_csv=Path(data["smiles_csv"]),
        output=Path(data["output"]),
        smiles_col=data.get("smiles_col") or None,
        id_prefix=data.get("id_prefix") or None,
        min_peaks=min_peaks,
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

    convert(cfg)


if __name__ == "__main__":
    main()
