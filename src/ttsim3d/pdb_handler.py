"""Handle PDB related operations."""

import mmdf
import torch


def load_model(file_path: str) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Pdb file to atom coordinates, ids, and B factors.

    Loads a pdb from `file_path` and returns the atom coordinates (in Angstroms),
    atom ids as a list of strings, and B factors (in Angstroms^2).

    Parameters
    ----------
    file_path : str
        Path to the pdb file.

    Returns
    -------
    tuple[torch.Tensor, list[str], torch.Tensor]
        Atom coordinates, atom ids, and B factors.
    """
    df = mmdf.read(file_path)
    atom_zyx = torch.tensor(df[["z", "y", "x"]].to_numpy()).float()  # (n_atoms, 3)
    atom_zyx -= torch.mean(atom_zyx, dim=0, keepdim=True)  # center
    atom_id = df["element"].str.upper().tolist()
    atom_b_factor = torch.tensor(df["b_isotropic"].to_numpy()).float()

    return atom_zyx, atom_id, atom_b_factor


def remove_hydrogens(
    atoms_zyx: torch.Tensor,
    atoms_id: list,
    atoms_b_factor_scaled: torch.Tensor,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """
    Remove hydrogen atoms from the atom list.

    Parameters
    ----------
    atoms_zyx : torch.Tensor
        Atom coordinates in Angstroms.
    atoms_id : list[str]
        Atom IDs.
    atoms_b_factor_scaled : torch.Tensor
        Atom B factors.

    Returns
    -------
    tuple[torch.Tensor, list[str], torch.Tensor]
        Atom coordinates, atom IDs, and B factors with hydrogen atoms removed.
    """
    non_h_mask = [aid != "H" for aid in atoms_id]
    atoms_zyx_filtered = atoms_zyx[non_h_mask]
    atoms_id_filtered = [aid for i, aid in enumerate(atoms_id) if non_h_mask[i]]
    atoms_b_factor_scaled_filtered = atoms_b_factor_scaled[non_h_mask]
    return atoms_zyx_filtered, atoms_id_filtered, atoms_b_factor_scaled_filtered
