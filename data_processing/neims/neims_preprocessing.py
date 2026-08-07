"""
neims_preprocessing.py — Build MIST training artifacts from a NEIMS MGF file.

The MGF produced by neims_output_to_mgf.py is the authoritative master record.
This script reads it and produces the four artifacts MIST needs:

  df_neims_DATASET_3_9_22.pkl     pickled DataFrame with columns SMILES, spec, name
  labels.tsv                      MIST-format labels (dataset, spec, ionization, ...)
  split_random.tsv                random train/val/test split       (split_mode: random)
  split_predefined.tsv            split derived from FEATURE_ID prefix (split_mode: predefined)
  split_all_test.tsv              every spectrum in test            (split_mode: all_test)
  DATASET_subforms_3_9_22.pkl     subformulae dict keyed by spectrum name
  subformulae/NAME.json           one subformula file per spectrum  (subform_format: json/both)

Inference datasets (unknown structures)
---------------------------------------
For experimental spectra whose molecules are unknown, the SMILES-derived fields
cannot be computed.  Such a dataset is preprocessed by:

  * omitting SMILES= from the MGF (parse_mgf tolerates its absence), and
  * pointing `labels_file` at a hand-built labels.tsv that supplies the
    externally determined `formula` (and `smiles`/`inchikey` placeholders).

When `labels_file` is set, labels.tsv is reused verbatim instead of being
derived from SMILES; only the row order is aligned to the MGF.  Use
`split_mode: all_test` and `subform_format: json` so the result is directly
loadable by MIST's PeakFormula featurizer, which globs `subform_folder/*.json`.

Usage:
  python neims_preprocessing.py config.yaml
  python neims_preprocessing.py --template > config.yaml
"""

import argparse
import dataclasses
import logging
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from pathlib import Path
from typing import Optional

import json

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

log = logging.getLogger(__name__)

# Column order MIST expects in labels.tsv.
_LABEL_COLS = [
    "dataset", "spec", "ionization", "formula", "smiles", "inchikey", "instrument",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    mgf_file: Path
    dataset_name: str
    output_dir: Path
    # reuse an existing labels.tsv instead of deriving one from SMILES
    labels_file: Optional[Path] = None
    # random split
    n_test: int = 16384
    n_val: int = 10240
    seed: int = 42
    # predefined split – derive label from FEATURE_ID prefix
    split_mode: str = "random"           # "random" | "predefined" | "all_test"
    train_id_prefix: Optional[str] = None
    val_id_prefix:   Optional[str] = None
    test_id_prefix:  Optional[str] = None
    # subformulae
    num_workers: int = 8
    mass_diff_thresh: float = 20.0
    max_formulae: int = 50
    subform_format: str = "pkl"          # "pkl" | "json" | "both"
    filter_above_parentmass: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        self.mgf_file = Path(self.mgf_file)
        self.output_dir = Path(self.output_dir)
        if self.labels_file is not None:
            self.labels_file = Path(self.labels_file)
        if self.split_mode not in ("random", "predefined", "all_test"):
            raise ValueError(
                "split_mode must be 'random', 'predefined' or 'all_test', "
                f"got {self.split_mode!r}"
            )
        if self.subform_format not in ("pkl", "json", "both"):
            raise ValueError(
                f"subform_format must be 'pkl', 'json' or 'both', got {self.subform_format!r}"
            )
        if self.split_mode == "predefined":
            missing = [k for k, v in [
                ("train_id_prefix", self.train_id_prefix),
                ("val_id_prefix",   self.val_id_prefix),
                ("test_id_prefix",  self.test_id_prefix),
            ] if not v]
            if missing:
                raise ValueError(
                    f"split_mode=predefined requires these keys: {missing}"
                )


_TEMPLATE = """\
# neims_preprocessing configuration
# Run with:  python neims_preprocessing.py config.yaml

# Path to the MGF file produced by neims_output_to_mgf.py (authoritative master record).
mgf_file: data/neims/my_dataset/spectra.mgf

# Short name for the dataset (used in filenames and the 'dataset' column of labels.tsv).
dataset_name: my_dataset

# Directory where all output files are written.
output_dir: data/neims/my_dataset/mist_inputs

# Existing labels.tsv to reuse verbatim instead of deriving one from SMILES.
# Required for inference datasets whose structures are unknown: the 'formula'
# column is then taken from this file rather than computed with RDKit.
# Leave null to derive labels from the SMILES in the MGF.
labels_file: null

# --- Split settings ---
# split_mode: random     -> shuffle all spectra, write split_random.tsv
# split_mode: predefined -> read split from FEATURE_ID prefix, write split_predefined.tsv
# split_mode: all_test   -> every spectrum to test, write split_all_test.tsv (inference)
split_mode: random
n_test: 16384   # used only when split_mode=random
n_val:  10240   # used only when split_mode=random
seed:   42      # used only when split_mode=random

# FEATURE_ID prefixes that identify each split (used only when split_mode=predefined).
# A spectrum whose name starts with train_id_prefix is assigned to train, etc.
train_id_prefix: null
val_id_prefix:   null
test_id_prefix:  null

# --- Subformulae settings ---
num_workers:      8
mass_diff_thresh: 20
max_formulae:     50

# How to write the subformula assignments:
#   pkl  -> one dict pickle (input to neims_dataset_builder.py)
#   json -> one NAME.json per spectrum under output_dir/subformulae/ (what
#           MIST's --subform-folder expects)
#   both -> write both
subform_format: pkl

# Drop peaks above PEPMASS + 1 before assigning subformulae, matching
# mist.utils.process_spec_file.  A no-op on NEIMS-simulated spectra (which never
# exceed the precursor); set true for experimental spectra, where above-precursor
# noise would otherwise consume the max_formulae peak budget.
filter_above_parentmass: false

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
    try:
        return float(s)
    except ValueError:
        pass
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def load_yaml_config(path: Path) -> Config:
    data: dict = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{lineno}: expected 'key: value', got: {line!r}")
            key, _, value = line.partition(":")
            data[key.strip()] = _parse_scalar(value)

    required = ("mgf_file", "dataset_name", "output_dir")
    missing = [k for k in required if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"Config file {path} is missing required keys: {missing}")

    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError(f"log_level must be DEBUG/INFO/WARNING/ERROR, got {log_level!r}")

    return Config(
        mgf_file=Path(data["mgf_file"]),
        dataset_name=str(data["dataset_name"]),
        output_dir=Path(data["output_dir"]),
        labels_file=Path(data["labels_file"]) if data.get("labels_file") else None,
        n_test=int(data.get("n_test", 16384)),
        n_val=int(data.get("n_val", 10240)),
        seed=int(data.get("seed", 42)),
        split_mode=str(data.get("split_mode", "random")),
        train_id_prefix=data.get("train_id_prefix") or None,
        val_id_prefix=data.get("val_id_prefix") or None,
        test_id_prefix=data.get("test_id_prefix") or None,
        num_workers=int(data.get("num_workers", 8)),
        mass_diff_thresh=float(data.get("mass_diff_thresh", 20.0)),
        max_formulae=int(data.get("max_formulae", 50)),
        subform_format=str(data.get("subform_format", "pkl")).lower(),
        filter_above_parentmass=bool(data.get("filter_above_parentmass", False)),
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# MGF parsing
# ---------------------------------------------------------------------------

def parse_mgf(mgf_path: Path) -> list[dict]:
    """
    Parse the MGF file produced by neims_output_to_mgf.py.

    Returns a list of dicts, each with:
      name       : FEATURE_ID value
      smiles     : SMILES string, or "" when the MGF carries no SMILES= line
                   (inference datasets, where the structure is unknown)
      spec       : (N, 2) float32 array of [mz, intensity] pairs
      parentmass : PEPMASS value as a float, or None when absent

    Only FEATURE_ID and at least one peak are required; a record missing either
    is dropped.
    """
    records = []
    current_meta: dict = {}
    current_peaks: list = []
    in_block = False
    n_dropped = 0
    n_no_smiles = 0

    with open(mgf_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line == "BEGIN IONS":
                in_block = True
                current_meta = {}
                current_peaks = []
            elif line == "END IONS":
                if current_peaks and "FEATURE_ID" in current_meta:
                    try:
                        parentmass = float(current_meta["PEPMASS"])
                    except (KeyError, ValueError):
                        parentmass = None
                    if "SMILES" not in current_meta:
                        n_no_smiles += 1
                    records.append({
                        "name": current_meta["FEATURE_ID"],
                        "smiles": current_meta.get("SMILES", ""),
                        "spec": np.array(current_peaks, dtype=np.float32),
                        "parentmass": parentmass,
                    })
                else:
                    n_dropped += 1
                in_block = False
            elif in_block:
                if "=" in line:
                    k, _, v = line.partition("=")
                    current_meta[k.strip()] = v.strip()
                elif line:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            current_peaks.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            pass

    log.info("Parsed %d spectra from %s", len(records), mgf_path.name)
    if n_dropped:
        log.warning("Dropped %d block(s) with no FEATURE_ID or no peaks", n_dropped)
    if n_no_smiles:
        log.info(
            "%d of %d spectra carry no SMILES= line — treating as an inference "
            "dataset (set labels_file to supply formulas).",
            n_no_smiles, len(records),
        )
    return records


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------

def _safe_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def _formula(mol) -> str:
    try:
        return CalcMolFormula(mol)
    except Exception:
        return ""


def _inchikey(mol) -> str:
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def build_dataframe(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame({
        "SMILES": [r["smiles"] for r in records],
        "spec":   [r["spec"]   for r in records],
        "name":   [r["name"]   for r in records],
    })


def build_labels(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    mols = df["SMILES"].apply(_safe_mol)
    return pd.DataFrame({
        "dataset":   f"{dataset_name}_neims",
        "spec":      df["name"].values,
        "ionization": "[M]+",
        "formula":   mols.apply(lambda m: _formula(m) if m else "").values,
        "smiles":    df["SMILES"].values,
        "inchikey":  mols.apply(lambda m: _inchikey(m) if m else "").values,
        "instrument": "simulated",
    })


def load_existing_labels(labels_path: Path, names: pd.Series) -> pd.DataFrame:
    """
    Load a hand-built labels.tsv and align it to the MGF's spectrum order.

    Used for inference datasets, where 'formula' comes from an external
    annotation rather than from RDKit.  The file is reused verbatim: no column
    is recomputed.  Every spectrum in the MGF must have a row; extra rows in
    the labels file are dropped with a warning.
    """
    labels = pd.read_csv(labels_path, sep="\t")
    labels = labels.drop(columns=[c for c in labels.columns if c.startswith("Unnamed")])

    missing_cols = [c for c in ("spec", "formula", "ionization") if c not in labels.columns]
    if missing_cols:
        raise ValueError(
            f"{labels_path} is missing required column(s) {missing_cols}. "
            f"Found: {list(labels.columns)}"
        )

    labels["spec"] = labels["spec"].astype(str)
    dupes = labels["spec"][labels["spec"].duplicated()].unique()
    if len(dupes):
        raise ValueError(
            f"{labels_path} has {len(dupes)} duplicated 'spec' value(s), "
            f"e.g. {list(dupes[:5])}. Each spectrum needs exactly one row."
        )

    by_name = labels.set_index("spec")
    wanted = [str(n) for n in names]
    missing = [n for n in wanted if n not in by_name.index]
    if missing:
        raise ValueError(
            f"{len(missing)} spectra in the MGF have no row in {labels_path}, "
            f"e.g. {missing[:5]}. Labels and MGF must describe the same spectra."
        )
    n_extra = len(by_name) - len(wanted)
    if n_extra > 0:
        log.warning(
            "%s has %d row(s) with no matching spectrum in the MGF — dropping them.",
            labels_path.name, n_extra,
        )

    out = by_name.loc[wanted].reset_index()
    for col in ("dataset", "smiles", "inchikey", "instrument"):
        if col not in out.columns:
            log.warning("%s has no '%s' column — filling with ''.", labels_path.name, col)
            out[col] = ""

    log.info("Reusing labels from %s (%d rows, aligned to MGF order)", labels_path, len(out))
    return out[_LABEL_COLS]


def build_split(names: pd.Series, n_test: int, n_val: int, seed: int) -> pd.DataFrame:
    n = len(names)
    n_test = min(n_test, n // 5)
    n_val  = min(n_val,  n // 10)
    n_train = n - n_test - n_val

    shuffled = names.sample(frac=1, random_state=seed).reset_index(drop=True)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return pd.DataFrame({"name": shuffled.values, "split": splits})


def build_split_predefined(
    names: pd.Series,
    train_prefix: str,
    val_prefix: str,
    test_prefix: str,
) -> pd.DataFrame:
    def _assign(name: str) -> str:
        if name.startswith(train_prefix):
            return "train"
        if name.startswith(val_prefix):
            return "val"
        if name.startswith(test_prefix):
            return "test"
        log.warning("Name %r does not match any split prefix — assigning to train", name)
        return "train"

    return pd.DataFrame({"name": names.values, "split": names.apply(_assign).values})


def build_split_all_test(names: pd.Series) -> pd.DataFrame:
    """Assign every spectrum to test — the split for an inference-only dataset."""
    return pd.DataFrame({"name": names.values, "split": "test"})


# ---------------------------------------------------------------------------
# Subformulae helpers (inlined from mist.utils – no mist/torch dependency)
# ---------------------------------------------------------------------------

_P_TBL = Chem.GetPeriodicTable()
_ELECTRON_MASS = 0.00054858
_CHEM_FORMULA_RE = re.compile(r"([A-Z][a-z]*)([0-9]*)")

_VALID_ELEMENTS = [
    "C", "H", "As", "B", "Br", "Cl", "Co", "F", "Fe",
    "I", "K", "N", "Na", "O", "P", "S", "Se", "Si",
]
_VALID_MONO_MASSES = np.array(
    [_P_TBL.GetMostCommonIsotopeMass(el) for el in _VALID_ELEMENTS]
)
_ELEMENT_VECTORS = np.eye(len(_VALID_ELEMENTS))
_element_to_ind = {el: i for i, el in enumerate(_VALID_ELEMENTS)}
_element_to_pos = {el: _ELEMENT_VECTORS[i] for i, el in enumerate(_VALID_ELEMENTS)}
_ELEMENT_TO_MASS = dict(zip(_VALID_ELEMENTS, _VALID_MONO_MASSES))

_ION_LST = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M+H3N+H]+", "[M]+", "[M-H4O2+H]+"]
_ion_to_mass = {
    "[M+H]+":      _ELEMENT_TO_MASS["H"]  - _ELECTRON_MASS,
    "[M+Na]+":     _ELEMENT_TO_MASS["Na"] - _ELECTRON_MASS,
    "[M+K]+":      _ELEMENT_TO_MASS["K"]  - _ELECTRON_MASS,
    "[M-H2O+H]+":  -_ELEMENT_TO_MASS["O"] - _ELEMENT_TO_MASS["H"] - _ELECTRON_MASS,
    "[M+H3N+H]+":  _ELEMENT_TO_MASS["N"] + _ELEMENT_TO_MASS["H"] * 4 - _ELECTRON_MASS,
    "[M]+":        -_ELECTRON_MASS,
    "[M-H4O2+H]+": -_ELEMENT_TO_MASS["O"] * 2 - _ELEMENT_TO_MASS["H"] * 3 - _ELECTRON_MASS,
}

_rdbe_mult = np.zeros(len(_VALID_ELEMENTS))
for _el, _w in zip(["C", "N", "P", "H", "Cl", "Br", "I", "F"], [2, 1, 1, -1, -1, -1, -1, -1]):
    _rdbe_mult[_element_to_ind[_el]] = _w


def _formula_to_dense(formula: str) -> np.ndarray:
    parts = []
    for sym, num in _CHEM_FORMULA_RE.findall(formula):
        if sym not in _element_to_pos:
            continue
        n = 1 if num == "" else int(num)
        parts.append(np.repeat(_element_to_pos[sym].reshape(1, -1), n, axis=0))
    return np.vstack(parts).sum(0) if parts else np.zeros(len(_VALID_ELEMENTS))


def _vec_to_formula(vec: np.ndarray) -> str:
    out = ""
    for i in np.argwhere(vec > 0).flatten():
        ct = int(vec[i])
        out += _VALID_ELEMENTS[i] + (str(ct) if ct > 1 else "")
    return out


def _cross_sum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.expand_dims(x, 0) + np.expand_dims(y, 1)).reshape(-1, y.shape[-1])


def _get_all_subsets(formula: str):
    dense = _formula_to_dense(formula)
    non_zero = np.argwhere(dense > 0).flatten()
    parts = [
        _ELEMENT_VECTORS[i] * np.arange(0, dense[i] + 1).reshape(-1, 1)
        for i in non_zero
    ]
    cross_prod = reduce(_cross_sum, parts, np.zeros((1, len(_VALID_ELEMENTS))))
    rdbe = 1 + 0.5 * cross_prod.dot(_rdbe_mult)
    cross_prod = cross_prod[rdbe >= 0]
    return cross_prod, cross_prod.dot(_VALID_MONO_MASSES)


def _clipped_ppm(diff: np.ndarray, mz: np.ndarray) -> np.ndarray:
    denom = mz.copy()
    denom[denom < 200] = 200
    return diff / denom * 1e6


def _process_spec(
    spec: np.ndarray,
    parentmass: float = 1e6,
    precision: int = 4,
    max_inten: float = 0.001,
    max_peaks: int = 60,
) -> Optional[np.ndarray]:
    """Normalise and filter a raw (mz, intensity) spectrum the same way mist does."""
    spec = spec[spec[:, 0] <= (parentmass + 1)].astype(np.float64)
    if spec.size == 0:
        return None
    # merge duplicate m/z (rounded), keep max intensity
    best: dict = {}
    for mz, inten in spec:
        key = round(mz, precision)
        if key not in best or inten > best[key][1]:
            best[key] = (mz, inten)
    spec = np.array(list(best.values()))
    spec[:, 1] /= spec[:, 1].max()
    spec[:, 1] = np.sqrt(spec[:, 1])
    # intensity threshold + top-k filter
    spec = spec[spec[:, 1] >= max_inten]
    if len(spec) > max_peaks:
        spec = spec[np.argsort(spec[:, 1])[-max_peaks:]]
    return spec if len(spec) > 0 else None


def _assign_one(args):
    """Top-level worker so ProcessPoolExecutor can pickle it."""
    name, spec_arr, formula, ion_type, mass_diff_thresh, max_formulae, parentmass = args
    spec = _process_spec(
        spec_arr,
        parentmass=1e6 if parentmass is None else parentmass,
        max_peaks=max_formulae,
    )
    result = {"cand_form": formula, "cand_ion": ion_type, "output_tbl": None}
    if spec is None or ion_type not in _ION_LST or not formula:
        return name, result

    cross_prod, masses = _get_all_subsets(formula)
    spec_mz, spec_int = spec[:, 0], spec[:, 1].copy()
    masses_with_ion = masses + _ion_to_mass[ion_type]

    diffs = np.abs(spec_mz[:, None] - masses_with_ion[None, :])
    fi = diffs.argmin(-1)
    min_diff = diffs[np.arange(len(diffs)), fi]
    ppm = _clipped_ppm(min_diff, spec_mz)

    valid = ppm < mass_diff_thresh
    spec_mz, spec_int = spec_mz[valid], spec_int[valid]
    min_diff, ppm, fi = min_diff[valid], ppm[valid], fi[valid]

    formulas = np.array([_vec_to_formula(cross_prod[i]) for i in fi])
    formula_masses = masses_with_ion[fi]

    seen: dict = {}
    uniq = []
    for idx, f in enumerate(formulas):
        if f not in seen:
            seen[f] = idx
            uniq.append(True)
        else:
            spec_int[seen[f]] += spec_int[idx]
            uniq.append(False)
    mask = np.array(uniq)

    if mask.any():
        result["output_tbl"] = {
            "mz":            list(spec_mz[mask]),
            "ms2_inten":     list(spec_int[mask]),
            "mono_mass":     list(formula_masses[mask]),
            "abs_mass_diff": list(min_diff[mask]),
            "mass_diff":     list(ppm[mask]),
            "formula":       list(formulas[mask]),
            "ions":          [ion_type] * int(mask.sum()),
        }
    return name, result


# ---------------------------------------------------------------------------
# Subformulae
# ---------------------------------------------------------------------------

def build_subforms_pkl(
    records: list,
    labels: pd.DataFrame,
    num_workers: int,
    mass_diff_thresh: float,
    max_formulae: int,
    filter_above_parentmass: bool = False,
) -> dict:
    formula_lookup = dict(zip(labels["spec"], labels["formula"]))
    ion_lookup = dict(zip(labels["spec"], labels["ionization"]))

    tasks = [
        (r["name"], r["spec"], formula_lookup.get(r["name"], ""),
         ion_lookup.get(r["name"], "[M]+"), mass_diff_thresh, max_formulae,
         r.get("parentmass") if filter_above_parentmass else None)
        for r in records
    ]

    subforms: dict = {}
    log.info("Assigning subformulae for %d spectra (workers=%d)…", len(tasks), num_workers)

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_assign_one, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                name, result = fut.result()
                subforms[name] = result
    else:
        for task in tasks:
            name, result = _assign_one(task)
            subforms[name] = result

    n_empty = sum(1 for v in subforms.values() if v["output_tbl"] is None)
    log.info("Collected %d subformulae entries (%d with no assigned peaks)",
             len(subforms), n_empty)
    return subforms


def write_subform_jsons(subforms: dict, output_dir: Path) -> Path:
    """
    Write one NAME.json per spectrum, the layout MIST's --subform-folder expects.

    Mirrors mist/subformulae/assign_subformulae.py: each file holds the
    cand_form / cand_ion / output_tbl dict for a single spectrum.
    """
    subform_dir = output_dir / "subformulae"
    subform_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in subforms.items():
        with open(subform_dir / f"{name}.json", "w") as fh:
            json.dump(entry, fh, indent=4)
    log.info("Wrote %d subformula JSON files to %s", len(subforms), subform_dir)
    return subform_dir


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.mgf_file.exists():
        raise FileNotFoundError(f"MGF file not found: {cfg.mgf_file}")

    ds = cfg.dataset_name

    # ------------------------------------------------------------------
    # 1. Parse MGF
    # ------------------------------------------------------------------
    records = parse_mgf(cfg.mgf_file)
    if not records:
        raise RuntimeError(f"No spectra parsed from {cfg.mgf_file}")

    # ------------------------------------------------------------------
    # 2. Resolve labels
    #
    # An inference dataset has no SMILES to derive formulas from, so its
    # labels are supplied externally and only re-ordered to match the MGF.
    # ------------------------------------------------------------------
    df = build_dataframe(records)

    if cfg.labels_file is not None:
        if not cfg.labels_file.exists():
            raise FileNotFoundError(f"labels_file not found: {cfg.labels_file}")
        labels = load_existing_labels(cfg.labels_file, df["name"])
        # Carry the supplied SMILES placeholders into the DataFrame so the two
        # artifacts agree on what is known about each molecule.
        df["SMILES"] = labels["smiles"].values
    else:
        labels = build_labels(df, ds)

    # ------------------------------------------------------------------
    # 3. DataFrame pickle and labels.tsv
    # ------------------------------------------------------------------
    pkl_path = cfg.output_dir / f"df_neims_{ds}_3_9_22.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(df, fh)
    log.info("Wrote %s (%d rows)", pkl_path.name, len(df))

    labels_path = cfg.output_dir / "labels.tsv"
    if cfg.labels_file is not None and cfg.labels_file.resolve() == labels_path.resolve():
        log.info("labels_file is the output labels.tsv — leaving it untouched.")
    else:
        labels.to_csv(labels_path, sep="\t", index=True)
        log.info("Wrote %s", labels_path.name)

    # ------------------------------------------------------------------
    # 4. split TSV
    # ------------------------------------------------------------------
    if cfg.split_mode == "all_test":
        split = build_split_all_test(df["name"])
        split_path = cfg.output_dir / "split_all_test.tsv"
    elif cfg.split_mode == "predefined":
        split = build_split_predefined(
            df["name"],
            cfg.train_id_prefix,
            cfg.val_id_prefix,
            cfg.test_id_prefix,
        )
        split_path = cfg.output_dir / "split_predefined.tsv"
    else:
        split = build_split(df["name"], cfg.n_test, cfg.n_val, cfg.seed)
        split_path = cfg.output_dir / "split_random.tsv"
    split.to_csv(split_path, sep="\t", index=True)
    log.info("Wrote %s  (train %d / val %d / test %d)",
             split_path.name,
             (split["split"] == "train").sum(),
             (split["split"] == "val").sum(),
             (split["split"] == "test").sum())

    # ------------------------------------------------------------------
    # 5. Subformulae
    # ------------------------------------------------------------------
    subforms = build_subforms_pkl(
        records=records,
        labels=labels,
        num_workers=cfg.num_workers,
        mass_diff_thresh=cfg.mass_diff_thresh,
        max_formulae=cfg.max_formulae,
        filter_above_parentmass=cfg.filter_above_parentmass,
    )

    if cfg.subform_format in ("pkl", "both"):
        subforms_pkl_path = cfg.output_dir / f"{ds}_subforms_3_9_22.pkl"
        with open(subforms_pkl_path, "wb") as fh:
            pickle.dump(subforms, fh)
        log.info("Wrote %s (%d entries)", subforms_pkl_path.name, len(subforms))

    if cfg.subform_format in ("json", "both"):
        write_subform_jsons(subforms, cfg.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
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

    run(cfg)
    log.info("Done.")


if __name__ == "__main__":
    main()