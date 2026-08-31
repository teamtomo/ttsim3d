"""Load mmdf structures for ttsim3d using monorepo structure-manipulation APIs."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mmdf
import pandas as pd
import torch
from torch_structure_manipulation import annotate_bonding_environments


@dataclass
class StructureLoadOptions:
    """Configuration options for loading a structure into a DataFrame.

    Attributes
    ----------
    center_atoms : bool
        Whether to center the atoms. Default is True.
    center_atoms_by_mass : bool
        If True, uses center of mass. If False, uses geometric center.
        Default is False (geometric center).
    center_point : tuple[float, float, float] | None
        Target center point in ZYX order. If None, centers at origin.
        Default is None.
    include_hydrogens : bool
        Whether to include hydrogen atoms in bonded-environment strings.
        Default is True.
    load_bonded_environment : bool
        Whether to compute bonded environments and molecule types.
        Default is True.
    """

    center_atoms: bool = True
    center_atoms_by_mass: bool = False
    center_point: tuple[float, float, float] | None = None
    include_hydrogens: bool = True
    load_bonded_environment: bool = True

    def __post_init__(self) -> None:
        """Validate option consistency after dataclass initialization."""
        if not self.center_atoms:
            if self.center_atoms_by_mass:
                raise ValueError("center_atoms_by_mass=True requires center_atoms=True.")
            if self.center_point is not None:
                raise ValueError("center_point cannot be set when center_atoms=False.")
        if self.center_point is not None and len(self.center_point) != 3:
            raise ValueError("center_point must be a 3-tuple of floats.")


def _center_dataframe(
    df: pd.DataFrame,
    *,
    center_point: tuple[float, float, float] | None,
    use_center_of_mass: bool,
) -> pd.DataFrame:
    """Center structure coordinates, materializing mmdf views for torch."""
    if use_center_of_mass:
        raise NotImplementedError(
            "center_atoms_by_mass is not yet supported in ttsim3d structure loading."
        )

    coords_zyx = torch.tensor(
        df[["z", "y", "x"]].to_numpy(copy=True),
        dtype=torch.float32,
    )
    coords_zyx -= coords_zyx.mean(dim=0, keepdim=True)
    if center_point is not None:
        coords_zyx += torch.tensor(center_point, dtype=torch.float32)

    result = df.copy()
    result.loc[:, ["z", "y", "x"]] = coords_zyx.numpy()
    return result


def load_structure(
    file_path: str | pathlib.Path,
    options: StructureLoadOptions | None = None,
) -> pd.DataFrame:
    """Load a structure file and return an annotated mmdf-compatible DataFrame."""
    if options is None:
        options = StructureLoadOptions()

    df = mmdf.read(file_path).copy()
    # mmdf can expose reversed coordinate views; materialize explicit coordinate columns.
    df.loc[:, ["x", "y", "z"]] = df[["x", "y", "z"]].to_numpy(copy=True)
    df.loc[:, ["z", "y", "x"]] = df[["z", "y", "x"]].to_numpy(copy=True)

    if options.center_atoms:
        df = _center_dataframe(
            df,
            center_point=options.center_point,
            use_center_of_mass=options.center_atoms_by_mass,
        )

    if options.load_bonded_environment:
        df = annotate_bonding_environments(
            df, include_hydrogens=options.include_hydrogens
        )
    else:
        df = df.copy()
        df["bonded_environment"] = None
        df["molecule_type"] = None

    return df


def df_params_to_tensors(
    df: pd.DataFrame,
) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str] | None, list[str] | None]:
    """Convert a structure DataFrame to the tensors used by the simulator."""
    atom_zyx = torch.tensor(df[["z", "y", "x"]].to_numpy(copy=True)).float()
    atom_id = df["element"].str.upper().tolist()
    atom_b_factor = torch.tensor(df["b_isotropic"].to_numpy(copy=True)).float()
    atom_bonded_id = df["bonded_environment"].tolist()
    molecule_type = df["molecule_type"].tolist()
    return atom_zyx, atom_id, atom_b_factor, atom_bonded_id, molecule_type
