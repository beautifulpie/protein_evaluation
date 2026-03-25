"""Core metrics for binary protein complex evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.spatial import cKDTree

from .align import apply_transform, compute_rmsd, kabsch_superimpose
from .io_utils import ResidueKey, ResidueRecord, StructureRecord, format_residue_keys

if TYPE_CHECKING:  # pragma: no cover
    from Bio import Align

BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")
CA_ONLY_ATOMS: tuple[str, ...] = ("CA",)
VDW_RADII: dict[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "SE": 1.90,
}


@dataclass
class ResidueMatch:
    """Matched residue pair between prediction and ground truth."""

    pred: ResidueRecord
    gt: ResidueRecord
    side: str
    pred_chain_id: str
    gt_chain_id: str


@dataclass(frozen=True)
class ChainMappingResult:
    """Chain-level mapping metadata for one receptor or ligand side."""

    chain_mapping: list[tuple[str, str]]
    warnings: list[str]
    strategy: str
    used_positional_mapping: bool


@dataclass(frozen=True)
class ChainMatchDiagnostic:
    """Diagnostics for one matched chain pair."""

    pred_chain_id: str
    gt_chain_id: str
    pred_length: int
    gt_length: int
    length_difference: int
    matched_residue_count: int
    unmatched_gt_residue_count: int
    unmatched_pred_residue_count: int
    matched_gt_fraction: float
    matched_pred_fraction: float
    sequence_identity: float
    used_sequence_fallback: bool


@dataclass
class SideMatchResult:
    """Residue matches for one partner side of the binary complex."""

    side: str
    chain_mapping: list[tuple[str, str]]
    matched_pairs: list[ResidueMatch] = field(default_factory=list)
    chain_diagnostics: list[ChainMatchDiagnostic] = field(default_factory=list)
    total_gt_residues: int = 0
    total_pred_residues: int = 0
    total_gt_heavy_atoms: int = 0
    used_sequence_fallback: bool = False
    chain_mapping_strategy: str = "exact_chain_names"
    used_positional_chain_mapping: bool = False
    warnings: list[str] = field(default_factory=list)

    def gt_to_pred(self) -> dict[ResidueKey, ResidueRecord]:
        """Return a mapping from ground-truth residue key to predicted residue."""

        return {pair.gt.key: pair.pred for pair in self.matched_pairs}


@dataclass
class ComplexMatchResult:
    """Combined receptor and ligand match result."""

    receptor: SideMatchResult
    ligand: SideMatchResult

    @property
    def all_pairs(self) -> list[ResidueMatch]:
        return [*self.receptor.matched_pairs, *self.ligand.matched_pairs]

    @property
    def total_gt_residues(self) -> int:
        return self.receptor.total_gt_residues + self.ligand.total_gt_residues

    @property
    def total_gt_heavy_atoms(self) -> int:
        return self.receptor.total_gt_heavy_atoms + self.ligand.total_gt_heavy_atoms

    @property
    def used_sequence_fallback(self) -> bool:
        return self.receptor.used_sequence_fallback or self.ligand.used_sequence_fallback

    @property
    def warnings(self) -> list[str]:
        return [*self.receptor.warnings, *self.ligand.warnings]


def build_chain_mapping(pred_chain_ids: Iterable[str], gt_chain_ids: Iterable[str]) -> ChainMappingResult:
    """Build a chain mapping, preferring exact chain identifiers when possible."""

    pred = _deduplicate_chain_ids(pred_chain_ids)
    gt = _deduplicate_chain_ids(gt_chain_ids)
    warnings: list[str] = []

    if not pred:
        raise ValueError("Prediction chain list is empty.")
    if not gt:
        raise ValueError("Ground-truth chain list is empty.")

    if len(pred) == len(gt) and set(pred) == set(gt):
        return ChainMappingResult(
            chain_mapping=[(chain_id, chain_id) for chain_id in pred],
            warnings=warnings,
            strategy="exact_chain_names",
            used_positional_mapping=False,
        )

    mapping = list(zip(pred, gt))
    if len(pred) != len(gt):
        warnings.append(
            f"Chain count mismatch for side mapping: pred={len(pred)} gt={len(gt)}. "
            "Only the first min(count) chains were mapped positionally."
        )
    elif pred != gt:
        warnings.append("Chain identifiers differ; mapped chains positionally according to manifest order.")
    return ChainMappingResult(
        chain_mapping=mapping,
        warnings=warnings,
        strategy="positional",
        used_positional_mapping=True,
    )


def match_complex_residues(
    pred_structure: StructureRecord,
    gt_structure: StructureRecord,
    pred_receptor_chains: list[str],
    pred_ligand_chains: list[str],
    gt_receptor_chains: list[str],
    gt_ligand_chains: list[str],
    allow_sequence_fallback: bool = False,
) -> ComplexMatchResult:
    """Match residues for receptor and ligand sides."""

    receptor = match_side_residues(
        pred_structure=pred_structure,
        gt_structure=gt_structure,
        pred_chain_ids=pred_receptor_chains,
        gt_chain_ids=gt_receptor_chains,
        side="receptor",
        allow_sequence_fallback=allow_sequence_fallback,
    )
    ligand = match_side_residues(
        pred_structure=pred_structure,
        gt_structure=gt_structure,
        pred_chain_ids=pred_ligand_chains,
        gt_chain_ids=gt_ligand_chains,
        side="ligand",
        allow_sequence_fallback=allow_sequence_fallback,
    )
    return ComplexMatchResult(receptor=receptor, ligand=ligand)


def match_side_residues(
    pred_structure: StructureRecord,
    gt_structure: StructureRecord,
    pred_chain_ids: list[str],
    gt_chain_ids: list[str],
    side: str,
    allow_sequence_fallback: bool = False,
) -> SideMatchResult:
    """Match residues for a single partner side."""

    chain_mapping_result = build_chain_mapping(pred_chain_ids, gt_chain_ids)
    warnings = list(chain_mapping_result.warnings)
    total_gt_residues = len(gt_structure.get_residues(gt_chain_ids))
    total_pred_residues = len(pred_structure.get_residues(pred_chain_ids))
    total_gt_heavy_atoms = sum(len(residue.heavy_atoms()) for residue in gt_structure.get_residues(gt_chain_ids))

    matched_pairs: list[ResidueMatch] = []
    chain_diagnostics: list[ChainMatchDiagnostic] = []
    used_sequence_fallback = False

    for pred_chain_id, gt_chain_id in chain_mapping_result.chain_mapping:
        pred_chain = pred_structure.get_chain(pred_chain_id)
        gt_chain = gt_structure.get_chain(gt_chain_id)
        if pred_chain is None:
            warnings.append(f"Prediction chain {pred_chain_id!r} was not found for {side}.")
            continue
        if gt_chain is None:
            warnings.append(f"Ground-truth chain {gt_chain_id!r} was not found for {side}.")
            continue

        exact_pairs = _match_chain_exact(pred_chain.residues, gt_chain.residues)
        sequence_pairs: list[tuple[ResidueRecord, ResidueRecord]] = []
        chosen_pairs = exact_pairs

        if allow_sequence_fallback:
            sequence_pairs = _match_chain_by_sequence(pred_chain.residues, gt_chain.residues)
            if _should_use_sequence_pairs(
                exact_pairs=exact_pairs,
                sequence_pairs=sequence_pairs,
                pred_residues=pred_chain.residues,
                gt_residues=gt_chain.residues,
            ):
                chosen_pairs = sequence_pairs
                used_sequence_fallback = True
                warnings.append(
                    f"Used sequence-based residue matching for {side} chain pair {pred_chain_id}->{gt_chain_id}."
                )

        chain_diagnostics.append(
            _build_chain_match_diagnostic(
                pred_chain_id=pred_chain_id,
                gt_chain_id=gt_chain_id,
                pred_residues=pred_chain.residues,
                gt_residues=gt_chain.residues,
                chosen_pairs=chosen_pairs,
                used_sequence_fallback=chosen_pairs is sequence_pairs and bool(sequence_pairs),
            )
        )
        for pred_residue, gt_residue in chosen_pairs:
            matched_pairs.append(
                ResidueMatch(
                    pred=pred_residue,
                    gt=gt_residue,
                    side=side,
                    pred_chain_id=pred_chain_id,
                    gt_chain_id=gt_chain_id,
                )
            )

    return SideMatchResult(
        side=side,
        chain_mapping=chain_mapping_result.chain_mapping,
        matched_pairs=matched_pairs,
        chain_diagnostics=chain_diagnostics,
        total_gt_residues=total_gt_residues,
        total_pred_residues=total_pred_residues,
        total_gt_heavy_atoms=total_gt_heavy_atoms,
        used_sequence_fallback=used_sequence_fallback,
        chain_mapping_strategy=chain_mapping_result.strategy,
        used_positional_chain_mapping=chain_mapping_result.used_positional_mapping,
        warnings=warnings,
    )


def compute_global_ca_rmsd(match_result: ComplexMatchResult) -> float:
    """Compute global CA RMSD across all matched residues."""

    pred_coords, gt_coords = _collect_named_atom_coordinates(match_result.all_pairs, atom_names=CA_ONLY_ATOMS)
    if len(pred_coords) == 0:
        return math.nan
    return kabsch_superimpose(pred_coords, gt_coords).rmsd


def compute_all_heavy_atom_rmsd(match_result: ComplexMatchResult) -> tuple[float, int]:
    """Compute all-heavy-atom RMSD across matched residues."""

    pred_coords, gt_coords, matched_count = _collect_heavy_atom_coordinates(match_result.all_pairs)
    if matched_count == 0:
        return math.nan, 0
    return kabsch_superimpose(pred_coords, gt_coords).rmsd, matched_count


def detect_interface_residues(
    receptor_residues: Iterable[ResidueRecord],
    ligand_residues: Iterable[ResidueRecord],
    cutoff: float = 10.0,
) -> tuple[set[ResidueKey], set[ResidueKey]]:
    """Return interface residue sets defined on the native complex."""

    receptor_coords, receptor_owner = _collect_heavy_atoms_with_owner(receptor_residues)
    ligand_coords, ligand_owner = _collect_heavy_atoms_with_owner(ligand_residues)
    if len(receptor_coords) == 0 or len(ligand_coords) == 0:
        return set(), set()

    tree = cKDTree(ligand_coords)
    receptor_interface: set[ResidueKey] = set()
    ligand_interface: set[ResidueKey] = set()

    for atom_index, neighbors in enumerate(tree.query_ball_point(receptor_coords, cutoff)):
        if not neighbors:
            continue
        receptor_interface.add(receptor_owner[atom_index])
        for ligand_index in neighbors:
            ligand_interface.add(ligand_owner[ligand_index])

    return receptor_interface, ligand_interface


def collect_interface_contacts(
    receptor_residues: Iterable[ResidueRecord],
    ligand_residues: Iterable[ResidueRecord],
    cutoff: float = 5.0,
) -> set[tuple[ResidueKey, ResidueKey]]:
    """Return residue-residue contacts across receptor and ligand."""

    receptor_coords, receptor_owner = _collect_heavy_atoms_with_owner(receptor_residues)
    ligand_coords, ligand_owner = _collect_heavy_atoms_with_owner(ligand_residues)
    if len(receptor_coords) == 0 or len(ligand_coords) == 0:
        return set()

    receptor_tree = cKDTree(receptor_coords)
    ligand_tree = cKDTree(ligand_coords)
    atom_pairs = receptor_tree.query_ball_tree(ligand_tree, cutoff)

    contacts: set[tuple[ResidueKey, ResidueKey]] = set()
    for receptor_index, neighbors in enumerate(atom_pairs):
        receptor_key = receptor_owner[receptor_index]
        for ligand_index in neighbors:
            contacts.add((receptor_key, ligand_owner[ligand_index]))
    return contacts


def compute_irmsd(
    match_result: ComplexMatchResult,
    native_receptor_interface: set[ResidueKey],
    native_ligand_interface: set[ResidueKey],
) -> tuple[float, bool]:
    """Compute interface RMSD using interface backbone atoms."""

    interface_pairs = [
        pair
        for pair in match_result.all_pairs
        if (pair.side == "receptor" and pair.gt.key in native_receptor_interface)
        or (pair.side == "ligand" and pair.gt.key in native_ligand_interface)
    ]
    return _compute_backbone_rmsd(interface_pairs)


def compute_lrmsd(match_result: ComplexMatchResult) -> tuple[float, bool]:
    """Compute ligand RMSD after fitting the receptor."""

    receptor_result = _collect_backbone_coordinates(match_result.receptor.matched_pairs)
    ligand_result = _collect_backbone_coordinates(match_result.ligand.matched_pairs)

    use_ca = receptor_result.use_ca_fallback or ligand_result.use_ca_fallback
    receptor_pred = receptor_result.ca_pred if use_ca else receptor_result.preferred_pred
    receptor_gt = receptor_result.ca_gt if use_ca else receptor_result.preferred_gt
    ligand_pred = ligand_result.ca_pred if use_ca else ligand_result.preferred_pred
    ligand_gt = ligand_result.ca_gt if use_ca else ligand_result.preferred_gt

    if len(receptor_pred) == 0 or len(receptor_gt) == 0 or len(ligand_pred) == 0 or len(ligand_gt) == 0:
        return math.nan, use_ca

    fit = kabsch_superimpose(receptor_pred, receptor_gt)
    transformed_ligand = apply_transform(ligand_pred, fit.rotation, fit.translation)
    return compute_rmsd(ligand_gt, transformed_ligand), use_ca


def compute_fnat(
    native_contacts: set[tuple[ResidueKey, ResidueKey]],
    match_result: ComplexMatchResult,
    contact_cutoff: float = 5.0,
) -> tuple[float, int, int]:
    """Compute the fraction of native contacts recovered by the predicted complex."""

    if not native_contacts:
        return math.nan, 0, 0

    receptor_map = match_result.receptor.gt_to_pred()
    ligand_map = match_result.ligand.gt_to_pred()

    recovered = 0
    for receptor_key, ligand_key in native_contacts:
        pred_receptor = receptor_map.get(receptor_key)
        pred_ligand = ligand_map.get(ligand_key)
        if pred_receptor is None or pred_ligand is None:
            continue
        if residue_contact_exists(pred_receptor, pred_ligand, cutoff=contact_cutoff):
            recovered += 1

    return recovered / len(native_contacts), len(native_contacts), recovered


def collect_predicted_contacts(
    match_result: ComplexMatchResult,
    contact_cutoff: float = 5.0,
) -> set[tuple[ResidueKey, ResidueKey]]:
    """Collect predicted residue-residue contacts projected into native residue key space.

    Only matched residues are included so the resulting contact set is comparable
    to the native contact set. This makes the diagnostic counts interpretable for
    benchmarking without silently conflating unmatched predicted residues with
    native residue identifiers.
    """

    receptor_coords, receptor_owner = _collect_mapped_heavy_atoms(match_result.receptor.matched_pairs)
    ligand_coords, ligand_owner = _collect_mapped_heavy_atoms(match_result.ligand.matched_pairs)
    if len(receptor_coords) == 0 or len(ligand_coords) == 0:
        return set()

    receptor_tree = cKDTree(receptor_coords)
    ligand_tree = cKDTree(ligand_coords)
    atom_pairs = receptor_tree.query_ball_tree(ligand_tree, contact_cutoff)

    contacts: set[tuple[ResidueKey, ResidueKey]] = set()
    for receptor_index, neighbors in enumerate(atom_pairs):
        receptor_key = receptor_owner[receptor_index]
        for ligand_index in neighbors:
            contacts.add((receptor_key, ligand_owner[ligand_index]))
    return contacts


def interface_contact_metrics(
    native_contacts: set[tuple[ResidueKey, ResidueKey]],
    predicted_contacts: set[tuple[ResidueKey, ResidueKey]],
) -> dict[str, float | int]:
    """Return explainable contact-set diagnostics.

    Precision/recall/F1 are left as NaN when their denominator is zero so that
    empty-interface edge cases are visible rather than silently scored as 0 or 1.
    """

    recovered_contacts = native_contacts & predicted_contacts
    missing_contacts = native_contacts - predicted_contacts
    false_contacts = predicted_contacts - native_contacts

    native_count = len(native_contacts)
    predicted_count = len(predicted_contacts)
    recovered_count = len(recovered_contacts)
    missing_count = len(missing_contacts)
    false_count = len(false_contacts)

    precision = recovered_count / predicted_count if predicted_count > 0 else math.nan
    recall = recovered_count / native_count if native_count > 0 else math.nan
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0.0:
        f1 = math.nan
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "interface_native_contact_count": native_count,
        "interface_pred_contact_count": predicted_count,
        "interface_recovered_contact_count": recovered_count,
        "interface_missing_contact_count": missing_count,
        "interface_false_contact_count": false_count,
        "interface_precision": precision,
        "interface_recall": recall,
        "interface_f1": f1,
    }


def dockq_score(fnat: float, irmsd: float, lrmsd: float) -> float:
    """Compute DockQ from Fnat, iRMSD, and LRMSD."""

    if any(math.isnan(value) for value in (fnat, irmsd, lrmsd)):
        return math.nan
    irmsd_term = 1.0 / (1.0 + (irmsd / 1.5) ** 2)
    lrmsd_term = 1.0 / (1.0 + (lrmsd / 8.5) ** 2)
    return (fnat + irmsd_term + lrmsd_term) / 3.0


def dockq_decomposition(fnat: float, irmsd: float, lrmsd: float) -> dict[str, float]:
    """Return the three DockQ decomposition terms."""

    irmsd_term = math.nan if math.isnan(irmsd) else 1.0 / (1.0 + (irmsd / 1.5) ** 2)
    lrmsd_term = math.nan if math.isnan(lrmsd) else 1.0 / (1.0 + (lrmsd / 8.5) ** 2)
    return {
        "dockq_decomposition_fnat_term": fnat,
        "dockq_decomposition_irmsd_term": irmsd_term,
        "dockq_decomposition_lrmsd_term": lrmsd_term,
    }


def dockq_success_flags(score: float) -> dict[str, bool]:
    """Return DockQ success categories."""

    if math.isnan(score):
        return {"acceptable": False, "medium": False, "high": False}
    return {
        "acceptable": score >= 0.23,
        "medium": score >= 0.49,
        "high": score >= 0.80,
    }


def compute_lddt_ca(match_result: ComplexMatchResult, cutoff: float = 15.0) -> float:
    """Compute CA-based lDDT on matched residues."""

    pred_coords, gt_coords = _collect_named_atom_coordinates(match_result.all_pairs, atom_names=CA_ONLY_ATOMS)
    if len(pred_coords) < 2:
        return math.nan

    gt_tree = cKDTree(gt_coords)
    native_pairs = list(gt_tree.query_pairs(cutoff))
    if not native_pairs:
        return math.nan

    scores: list[float] = []
    for left, right in native_pairs:
        native_distance = float(np.linalg.norm(gt_coords[left] - gt_coords[right]))
        pred_distance = float(np.linalg.norm(pred_coords[left] - pred_coords[right]))
        error = abs(pred_distance - native_distance)
        score = sum(error <= threshold for threshold in (0.5, 1.0, 2.0, 4.0)) / 4.0
        scores.append(score)

    return float(np.mean(scores))


def count_steric_clashes(
    residues: Iterable[ResidueRecord],
    overlap_tolerance: float = 0.4,
) -> tuple[int, float]:
    """Count approximate steric clashes among heavy atoms."""

    atom_records: list[tuple[np.ndarray, ResidueRecord, str, str]] = []
    for residue in residues:
        for atom in residue.heavy_atoms():
            atom_records.append((atom.coord, residue, atom.name, atom.element))

    if len(atom_records) < 2:
        return 0, 0.0

    coords = np.vstack([record[0] for record in atom_records])
    max_radius = max(VDW_RADII.values())
    tree = cKDTree(coords)
    candidate_pairs = tree.query_pairs(2.0 * max_radius)

    clash_count = 0
    for left, right in candidate_pairs:
        left_coord, left_residue, left_name, left_element = atom_records[left]
        right_coord, right_residue, right_name, right_element = atom_records[right]
        if left_residue.key == right_residue.key:
            continue
        if _is_obvious_bonded_neighbor(left_residue, left_name, right_residue, right_name):
            continue

        radius_sum = VDW_RADII.get(left_element, 1.7) + VDW_RADII.get(right_element, 1.7)
        distance = float(np.linalg.norm(left_coord - right_coord))
        if distance < radius_sum - overlap_tolerance:
            clash_count += 1

    clashes_per_1000_atoms = 1000.0 * clash_count / len(atom_records)
    return clash_count, clashes_per_1000_atoms


def evaluate_binary_complex(
    pred_structure: StructureRecord,
    gt_structure: StructureRecord,
    pred_receptor_chains: list[str],
    pred_ligand_chains: list[str],
    gt_receptor_chains: list[str],
    gt_ligand_chains: list[str],
    contact_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    include_all_atom_rmsd: bool = True,
    include_lddt: bool = True,
    include_clashes: bool = True,
    sequence_fallback: bool = False,
) -> dict[str, object]:
    """Evaluate one predicted complex against its ground-truth structure."""

    match_result = match_complex_residues(
        pred_structure=pred_structure,
        gt_structure=gt_structure,
        pred_receptor_chains=pred_receptor_chains,
        pred_ligand_chains=pred_ligand_chains,
        gt_receptor_chains=gt_receptor_chains,
        gt_ligand_chains=gt_ligand_chains,
        allow_sequence_fallback=sequence_fallback,
    )

    gt_receptor_residues = gt_structure.get_residues(gt_receptor_chains)
    gt_ligand_residues = gt_structure.get_residues(gt_ligand_chains)
    pred_all_residues = pred_structure.get_residues([*pred_receptor_chains, *pred_ligand_chains])

    num_matched_residues = len(match_result.all_pairs)
    matched_residue_fraction = num_matched_residues / max(1, match_result.total_gt_residues)

    heavy_rmsd, num_matched_atoms = compute_all_heavy_atom_rmsd(match_result)
    matched_atom_fraction = num_matched_atoms / max(1, match_result.total_gt_heavy_atoms)

    native_interface_receptor, native_interface_ligand = detect_interface_residues(
        receptor_residues=gt_receptor_residues,
        ligand_residues=gt_ligand_residues,
        cutoff=interface_cutoff,
    )
    irmsd, used_ca_fallback_for_irmsd = compute_irmsd(
        match_result=match_result,
        native_receptor_interface=native_interface_receptor,
        native_ligand_interface=native_interface_ligand,
    )
    lrmsd, used_ca_fallback_for_lrmsd = compute_lrmsd(match_result)

    native_contacts = collect_interface_contacts(
        receptor_residues=gt_receptor_residues,
        ligand_residues=gt_ligand_residues,
        cutoff=contact_cutoff,
    )
    predicted_contacts = collect_predicted_contacts(match_result, contact_cutoff=contact_cutoff)
    fnat, num_native_contacts, num_recovered_contacts = compute_fnat(
        native_contacts=native_contacts,
        match_result=match_result,
        contact_cutoff=contact_cutoff,
    )

    dockq = dockq_score(fnat, irmsd, lrmsd)
    dockq_terms = dockq_decomposition(fnat, irmsd, lrmsd)
    success_flags = dockq_success_flags(dockq)
    lddt_ca = compute_lddt_ca(match_result) if include_lddt else math.nan
    interface_metrics = interface_contact_metrics(native_contacts, predicted_contacts)

    if include_clashes:
        clash_count, clashes_per_1000_atoms = count_steric_clashes(pred_all_residues)
    else:
        clash_count, clashes_per_1000_atoms = math.nan, math.nan

    metrics: dict[str, object] = {
        "ca_rmsd": compute_global_ca_rmsd(match_result),
        "all_atom_rmsd": heavy_rmsd if include_all_atom_rmsd else math.nan,
        "irmsd": irmsd,
        "lrmsd": lrmsd,
        "fnat": fnat,
        "dockq": dockq,
        "dockq_acceptable": success_flags["acceptable"],
        "dockq_medium": success_flags["medium"],
        "dockq_high": success_flags["high"],
        "lddt_ca": lddt_ca,
        "clash_count": clash_count,
        "clashes_per_1000_atoms": clashes_per_1000_atoms,
        "matched_residue_fraction": matched_residue_fraction,
        "matched_atom_fraction": matched_atom_fraction,
        "num_matched_residues": num_matched_residues,
        "num_matched_atoms": num_matched_atoms,
        "num_native_contacts": num_native_contacts,
        "num_recovered_contacts": num_recovered_contacts,
        "num_native_interface_receptor_residues": len(native_interface_receptor),
        "num_native_interface_ligand_residues": len(native_interface_ligand),
        "native_interface_receptor_residues": format_residue_keys(native_interface_receptor),
        "native_interface_ligand_residues": format_residue_keys(native_interface_ligand),
        "used_sequence_fallback": match_result.used_sequence_fallback,
        "used_ca_fallback_for_irmsd": used_ca_fallback_for_irmsd,
        "used_ca_fallback_for_lrmsd": used_ca_fallback_for_lrmsd,
        "parse_warning": _merge_warnings(pred_structure.warnings + gt_structure.warnings + match_result.warnings),
    }
    metrics.update(dockq_terms)
    metrics.update(interface_metrics)
    metrics.update(mapping_diagnostics(match_result))
    return metrics


def residue_contact_exists(residue_a: ResidueRecord, residue_b: ResidueRecord, cutoff: float = 5.0) -> bool:
    """Return whether two residues make a heavy-atom contact."""

    coords_a = np.vstack([atom.coord for atom in residue_a.heavy_atoms()]) if residue_a.heavy_atoms() else None
    coords_b = np.vstack([atom.coord for atom in residue_b.heavy_atoms()]) if residue_b.heavy_atoms() else None
    if coords_a is None or coords_b is None:
        return False
    squared_distances = np.sum((coords_a[:, None, :] - coords_b[None, :, :]) ** 2, axis=2)
    return bool(np.any(squared_distances <= cutoff * cutoff))


def _deduplicate_chain_ids(chain_ids: Iterable[str]) -> list[str]:
    """Return chain identifiers preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for chain_id in chain_ids:
        normalized = str(chain_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _match_chain_exact(
    pred_residues: list[ResidueRecord],
    gt_residues: list[ResidueRecord],
) -> list[tuple[ResidueRecord, ResidueRecord]]:
    """Match residues by residue number, insertion code, and residue name."""

    gt_lookup = {(res.resseq, res.insertion_code, res.resname): res for res in gt_residues}
    matches: list[tuple[ResidueRecord, ResidueRecord]] = []
    seen_gt: set[ResidueKey] = set()
    for pred_residue in pred_residues:
        key = (pred_residue.resseq, pred_residue.insertion_code, pred_residue.resname)
        gt_residue = gt_lookup.get(key)
        if gt_residue is None or gt_residue.key in seen_gt:
            continue
        matches.append((pred_residue, gt_residue))
        seen_gt.add(gt_residue.key)
    return matches


def _match_chain_by_sequence(
    pred_residues: list[ResidueRecord],
    gt_residues: list[ResidueRecord],
) -> list[tuple[ResidueRecord, ResidueRecord]]:
    """Match residues with global sequence alignment."""

    try:
        from Bio import Align
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "biopython is required for sequence-based residue matching. "
            "Install dependencies from requirements.txt or disable --sequence_fallback."
        ) from exc

    if not pred_residues or not gt_residues:
        return []

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5

    gt_sequence = "".join(residue.sequence_code for residue in gt_residues)
    pred_sequence = "".join(residue.sequence_code for residue in pred_residues)
    alignment = aligner.align(gt_sequence, pred_sequence)[0]

    matches: list[tuple[ResidueRecord, ResidueRecord]] = []
    for gt_block, pred_block in zip(alignment.aligned[0], alignment.aligned[1]):
        gt_start, gt_end = gt_block
        pred_start, pred_end = pred_block
        block_len = min(gt_end - gt_start, pred_end - pred_start)
        for offset in range(block_len):
            gt_residue = gt_residues[gt_start + offset]
            pred_residue = pred_residues[pred_start + offset]
            if (
                gt_residue.resname == pred_residue.resname
                or gt_residue.sequence_code == pred_residue.sequence_code
            ):
                matches.append((pred_residue, gt_residue))
    return matches


def _should_use_sequence_pairs(
    exact_pairs: list[tuple[ResidueRecord, ResidueRecord]],
    sequence_pairs: list[tuple[ResidueRecord, ResidueRecord]],
    pred_residues: list[ResidueRecord],
    gt_residues: list[ResidueRecord],
) -> bool:
    """Decide when sequence-based matching is preferable."""

    if not sequence_pairs:
        return False
    if not exact_pairs:
        return True
    denominator = max(1, min(len(pred_residues), len(gt_residues)))
    exact_fraction = len(exact_pairs) / denominator
    sequence_fraction = len(sequence_pairs) / denominator
    return exact_fraction < 0.8 and sequence_fraction > exact_fraction


def _build_chain_match_diagnostic(
    pred_chain_id: str,
    gt_chain_id: str,
    pred_residues: list[ResidueRecord],
    gt_residues: list[ResidueRecord],
    chosen_pairs: list[tuple[ResidueRecord, ResidueRecord]],
    used_sequence_fallback: bool,
) -> ChainMatchDiagnostic:
    """Summarize matching quality for one mapped chain pair."""

    matched_residue_count = len(chosen_pairs)
    identical_sequence_codes = sum(
        pred_residue.sequence_code == gt_residue.sequence_code
        for pred_residue, gt_residue in chosen_pairs
    )
    identity_denominator = max(1, max(len(pred_residues), len(gt_residues)))
    return ChainMatchDiagnostic(
        pred_chain_id=pred_chain_id,
        gt_chain_id=gt_chain_id,
        pred_length=len(pred_residues),
        gt_length=len(gt_residues),
        length_difference=len(pred_residues) - len(gt_residues),
        matched_residue_count=matched_residue_count,
        unmatched_gt_residue_count=max(0, len(gt_residues) - matched_residue_count),
        unmatched_pred_residue_count=max(0, len(pred_residues) - matched_residue_count),
        matched_gt_fraction=matched_residue_count / max(1, len(gt_residues)),
        matched_pred_fraction=matched_residue_count / max(1, len(pred_residues)),
        sequence_identity=identical_sequence_codes / identity_denominator,
        used_sequence_fallback=used_sequence_fallback,
    )


def mapping_diagnostics(match_result: ComplexMatchResult) -> dict[str, object]:
    """Return mapping diagnostics and aggregate quality indicators."""

    diagnostics: dict[str, object] = {}
    min_sequence_identity = 1.0
    if not match_result.all_pairs:
        min_sequence_identity = math.nan

    for side_name, side_result in (("receptor", match_result.receptor), ("ligand", match_result.ligand)):
        matched_count = len(side_result.matched_pairs)
        unmatched_gt = max(0, side_result.total_gt_residues - matched_count)
        unmatched_pred = max(0, side_result.total_pred_residues - matched_count)
        matched_fraction = matched_count / max(1, side_result.total_gt_residues)
        matched_pred_fraction = matched_count / max(1, side_result.total_pred_residues)
        chain_length_differences = [diagnostic.length_difference for diagnostic in side_result.chain_diagnostics]
        sequence_identities = [diagnostic.sequence_identity for diagnostic in side_result.chain_diagnostics]
        if sequence_identities:
            min_sequence_identity = min(min_sequence_identity, min(sequence_identities))

        diagnostics.update(
            {
                f"{side_name}_num_chain_pairs": len(side_result.chain_mapping),
                f"{side_name}_num_matched_residues": matched_count,
                f"{side_name}_num_unmatched_gt_residues": unmatched_gt,
                f"{side_name}_num_unmatched_pred_residues": unmatched_pred,
                f"{side_name}_matched_residue_fraction": matched_fraction,
                f"{side_name}_matched_pred_fraction": matched_pred_fraction,
                f"{side_name}_min_chain_sequence_identity": min(sequence_identities) if sequence_identities else math.nan,
                f"{side_name}_max_chain_length_difference": (
                    max(abs(value) for value in chain_length_differences) if chain_length_differences else math.nan
                ),
                f"{side_name}_chain_length_differences": _serialize_chain_length_differences(side_result.chain_diagnostics),
                f"{side_name}_used_sequence_fallback": side_result.used_sequence_fallback,
                f"{side_name}_used_positional_chain_mapping": side_result.used_positional_chain_mapping,
                f"{side_name}_chain_mapping_strategy": side_result.chain_mapping_strategy,
                f"{side_name}_chain_match_summary": _serialize_chain_match_summary(side_result.chain_diagnostics),
            }
        )

    confidence_floor = min(
        value
        for value in [
            diagnostics.get("receptor_matched_residue_fraction", math.nan),
            diagnostics.get("ligand_matched_residue_fraction", math.nan),
            min_sequence_identity,
        ]
        if not math.isnan(float(value))
    ) if match_result.all_pairs else math.nan
    diagnostics["mapping_confidence_score"] = confidence_floor
    diagnostics["mapping_confidence_floor_signal"] = confidence_floor
    return diagnostics


def _collect_named_atom_coordinates(
    matches: Iterable[ResidueMatch],
    atom_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect coordinates for a fixed list of atom names."""

    pred_coords: list[np.ndarray] = []
    gt_coords: list[np.ndarray] = []
    for match in matches:
        for atom_name in atom_names:
            pred_atom = match.pred.get_atom(atom_name)
            gt_atom = match.gt.get_atom(atom_name)
            if pred_atom is None or gt_atom is None:
                continue
            pred_coords.append(pred_atom.coord)
            gt_coords.append(gt_atom.coord)

    if not pred_coords:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    return np.vstack(pred_coords), np.vstack(gt_coords)


def _collect_heavy_atom_coordinates(matches: Iterable[ResidueMatch]) -> tuple[np.ndarray, np.ndarray, int]:
    """Collect matched heavy-atom coordinates by residue and atom name."""

    pred_coords: list[np.ndarray] = []
    gt_coords: list[np.ndarray] = []
    for match in matches:
        common_atom_names = sorted(set(match.pred.atoms) & set(match.gt.atoms))
        for atom_name in common_atom_names:
            pred_atom = match.pred.atoms[atom_name]
            gt_atom = match.gt.atoms[atom_name]
            if pred_atom.element in {"H", "D"} or gt_atom.element in {"H", "D"}:
                continue
            pred_coords.append(pred_atom.coord)
            gt_coords.append(gt_atom.coord)

    if not pred_coords:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty, 0
    return np.vstack(pred_coords), np.vstack(gt_coords), len(pred_coords)


@dataclass(frozen=True)
class BackboneCoordinateResult:
    """Preferred and CA-only backbone coordinate collections."""

    preferred_pred: np.ndarray
    preferred_gt: np.ndarray
    ca_pred: np.ndarray
    ca_gt: np.ndarray
    use_ca_fallback: bool


def _collect_backbone_coordinates(matches: Iterable[ResidueMatch]) -> BackboneCoordinateResult:
    """Collect preferred and fallback backbone coordinates."""

    preferred_pred, preferred_gt = _collect_named_atom_coordinates(matches, atom_names=BACKBONE_ATOMS)
    ca_pred, ca_gt = _collect_named_atom_coordinates(matches, atom_names=CA_ONLY_ATOMS)

    use_ca = False
    if len(preferred_pred) == 0 and len(ca_pred) > 0:
        use_ca = True
    elif len(ca_pred) > 0 and len(preferred_pred) < max(3, 2 * len(ca_pred)):
        use_ca = True

    return BackboneCoordinateResult(
        preferred_pred=preferred_pred,
        preferred_gt=preferred_gt,
        ca_pred=ca_pred,
        ca_gt=ca_gt,
        use_ca_fallback=use_ca,
    )


def _compute_backbone_rmsd(matches: Iterable[ResidueMatch]) -> tuple[float, bool]:
    """Compute backbone RMSD with CA fallback when backbone coverage is poor."""

    result = _collect_backbone_coordinates(matches)
    pred_coords = result.ca_pred if result.use_ca_fallback else result.preferred_pred
    gt_coords = result.ca_gt if result.use_ca_fallback else result.preferred_gt
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return math.nan, result.use_ca_fallback
    return kabsch_superimpose(pred_coords, gt_coords).rmsd, result.use_ca_fallback


def _collect_heavy_atoms_with_owner(
    residues: Iterable[ResidueRecord],
) -> tuple[np.ndarray, list[ResidueKey]]:
    """Collect heavy-atom coordinates together with the owning residue key."""

    coords: list[np.ndarray] = []
    owners: list[ResidueKey] = []
    for residue in residues:
        for atom in residue.heavy_atoms():
            coords.append(atom.coord)
            owners.append(residue.key)

    if not coords:
        return np.empty((0, 3), dtype=float), []
    return np.vstack(coords), owners


def _collect_mapped_heavy_atoms(
    matches: Iterable[ResidueMatch],
) -> tuple[np.ndarray, list[ResidueKey]]:
    """Collect predicted heavy atoms using ground-truth residue keys as owners."""

    coords: list[np.ndarray] = []
    owners: list[ResidueKey] = []
    for match in matches:
        for atom in match.pred.heavy_atoms():
            coords.append(atom.coord)
            owners.append(match.gt.key)

    if not coords:
        return np.empty((0, 3), dtype=float), []
    return np.vstack(coords), owners


def _is_obvious_bonded_neighbor(
    left_residue: ResidueRecord,
    left_atom_name: str,
    right_residue: ResidueRecord,
    right_atom_name: str,
) -> bool:
    """Exclude a minimal set of obvious bonded peptide neighbors."""

    same_chain = left_residue.chain_id == right_residue.chain_id
    adjacent = abs(left_residue.resseq - right_residue.resseq) == 1
    if not (same_chain and adjacent and left_residue.insertion_code == right_residue.insertion_code == ""):
        return False
    peptide_bond_pairs = {frozenset({"C", "N"})}
    return frozenset({left_atom_name, right_atom_name}) in peptide_bond_pairs


def _merge_warnings(warnings: Iterable[str]) -> str:
    """Merge warnings into a stable string."""

    unique: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        text = str(warning).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return " | ".join(unique)


def _serialize_chain_length_differences(chain_diagnostics: Iterable[ChainMatchDiagnostic]) -> str:
    """Serialize chain length deltas to a stable compact string."""

    return ";".join(
        f"{diagnostic.pred_chain_id}->{diagnostic.gt_chain_id}:{diagnostic.length_difference}"
        for diagnostic in chain_diagnostics
    )


def _serialize_chain_match_summary(chain_diagnostics: Iterable[ChainMatchDiagnostic]) -> str:
    """Serialize chain-level matching coverage and sequence identity."""

    return ";".join(
        (
            f"{diagnostic.pred_chain_id}->{diagnostic.gt_chain_id}"
            f":matched={diagnostic.matched_residue_count}"
            f"/gt={diagnostic.gt_length}"
            f"/pred={diagnostic.pred_length}"
            f"/seqid={diagnostic.sequence_identity:.3f}"
            f"/fallback={int(diagnostic.used_sequence_fallback)}"
        )
        for diagnostic in chain_diagnostics
    )
