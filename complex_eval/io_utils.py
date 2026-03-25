"""Structure parsing and lightweight internal data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import gemmi

AA3_TO_AA1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "ASX": "B",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLX": "Z",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
    "VAL": "V",
}


@dataclass(frozen=True)
class AtomRecord:
    """Compact atom representation used by metric calculations."""

    name: str
    element: str
    coord: np.ndarray
    occupancy: float
    altloc: str = ""


@dataclass(frozen=True)
class ResidueKey:
    """Identifier for a residue within a structure."""

    chain_id: str
    resseq: int
    insertion_code: str
    resname: str

    def to_string(self) -> str:
        """Return a stable human-readable residue identifier."""

        insertion = self.insertion_code or "."
        return f"{self.chain_id}:{self.resname}:{self.resseq}:{insertion}"


@dataclass
class ResidueRecord:
    """Residue representation with atom lookup helpers."""

    chain_id: str
    resseq: int
    insertion_code: str
    resname: str
    atoms: dict[str, AtomRecord]
    sequence_index: int

    @property
    def key(self) -> ResidueKey:
        return ResidueKey(
            chain_id=self.chain_id,
            resseq=self.resseq,
            insertion_code=self.insertion_code,
            resname=self.resname,
        )

    @property
    def sequence_code(self) -> str:
        return AA3_TO_AA1.get(self.resname.upper(), "X")

    def get_atom(self, atom_name: str) -> AtomRecord | None:
        return self.atoms.get(atom_name)

    def heavy_atoms(self) -> list[AtomRecord]:
        return [atom for atom in self.atoms.values() if atom.element not in {"H", "D"}]


@dataclass
class ChainRecord:
    """Protein chain containing ordered residues."""

    chain_id: str
    residues: list[ResidueRecord] = field(default_factory=list)

    @property
    def sequence(self) -> str:
        return "".join(residue.sequence_code for residue in self.residues)


@dataclass
class StructureRecord:
    """Parsed structure using the first model only."""

    path: Path
    chains: dict[str, ChainRecord]
    warnings: list[str] = field(default_factory=list)

    def get_chain(self, chain_id: str) -> ChainRecord | None:
        return self.chains.get(chain_id)

    def get_residues(self, chain_ids: Iterable[str]) -> list[ResidueRecord]:
        residues: list[ResidueRecord] = []
        for chain_id in chain_ids:
            chain = self.get_chain(chain_id)
            if chain is not None:
                residues.extend(chain.residues)
        return residues


def parse_chain_list(value: str | float | None) -> list[str]:
    """Parse a comma-separated chain list."""

    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_chain_groups(value: str | float | None) -> list[list[str]]:
    """Parse pipe-separated chain groups with comma-separated chain IDs."""

    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    groups = [parse_chain_list(group_text) for group_text in text.split("|")]
    return [group for group in groups if group]


def resolve_input_path(path_value: str | Path, base_dir: str | Path) -> Path:
    """Resolve an input path relative to the manifest directory."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def load_structure(path: str | Path, ignore_hydrogens: bool = True) -> StructureRecord:
    """Load a PDB or mmCIF structure into a consistent internal representation."""

    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "gemmi is required to parse structures. Install dependencies from requirements.txt."
        ) from exc

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Structure file does not exist: {path_obj}")

    try:
        structure = gemmi.read_structure(str(path_obj))
    except Exception as exc:  # pragma: no cover - gemmi message is informative
        raise ValueError(f"Failed to parse structure {path_obj}: {exc}") from exc

    if len(structure) == 0:
        raise ValueError(f"Structure {path_obj} does not contain any models.")

    model = structure[0]
    chains: dict[str, ChainRecord] = {}
    warnings: list[str] = []

    for chain in model:
        residues: list[ResidueRecord] = []
        sequence_index = 0
        for residue in chain:
            resname = residue.name.strip().upper()
            if resname not in AA3_TO_AA1:
                continue

            atoms = _select_atoms(residue, ignore_hydrogens=ignore_hydrogens)
            if not atoms:
                warnings.append(
                    f"Skipped residue without retained atoms: {chain.name}:{resname}:{residue.seqid.num}"
                )
                continue

            insertion_code = str(residue.seqid.icode).strip()
            if insertion_code in {".", "?", "0"}:
                insertion_code = ""

            residues.append(
                ResidueRecord(
                    chain_id=chain.name,
                    resseq=int(residue.seqid.num),
                    insertion_code=insertion_code,
                    resname=resname,
                    atoms=atoms,
                    sequence_index=sequence_index,
                )
            )
            sequence_index += 1

        if residues:
            chains[chain.name] = ChainRecord(chain_id=chain.name, residues=residues)

    if not chains:
        raise ValueError(f"No protein residues were found in {path_obj}.")

    return StructureRecord(path=path_obj.resolve(), chains=chains, warnings=warnings)


def format_residue_keys(keys: Iterable[ResidueKey]) -> str:
    """Serialize a residue key collection to a stable semicolon-separated string."""

    ordered = sorted(key.to_string() for key in keys)
    return ";".join(ordered)


def _select_atoms(residue: gemmi.Residue, ignore_hydrogens: bool) -> dict[str, AtomRecord]:
    """Select one atom per atom name, keeping the best altloc by occupancy."""

    selected: dict[str, tuple[tuple[float, int], AtomRecord]] = {}

    for atom in residue:
        atom_name = atom.name.strip().upper()
        if not atom_name:
            continue

        element = _atom_element(atom)
        if ignore_hydrogens and element in {"H", "D"}:
            continue

        altloc = _normalize_altloc(str(atom.altloc))
        occupancy = float(atom.occ)
        rank = (occupancy, 1 if not altloc else 0)
        record = AtomRecord(
            name=atom_name,
            element=element,
            coord=np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float),
            occupancy=occupancy,
            altloc=altloc,
        )

        current = selected.get(atom_name)
        if current is None or rank > current[0]:
            selected[atom_name] = (rank, record)

    return {name: record for name, (_, record) in selected.items()}


def _atom_element(atom: gemmi.Atom) -> str:
    """Return a normalized element symbol."""

    element = getattr(atom.element, "name", "") or ""
    element = str(element).strip().upper()
    if element:
        return element
    return atom.name.strip()[:1].upper()


def _normalize_altloc(value: str) -> str:
    """Normalize altloc markers to blank or a single identifier."""

    text = value.strip()
    if text in {"", ".", "?", "\x00"}:
        return ""
    return text
