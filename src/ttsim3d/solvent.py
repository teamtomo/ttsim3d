"""The solvent a molecule displaces: its solvent-excluded volume on the simulation grid.

ttsim3d simulates a molecule in vacuum. In a specimen the molecule displaces solvent,
whose own potential is missing from the vacuum map. Subtracting it gives the template

    A - rho * E,

where A is the simulated potential before any dose or MTF filter, E is the molecule's
solvent-excluded volume scaled to the mean of A inside it, and rho is the solvent's
potential as a fraction of that mean (about 0.6 for water against protein and RNA).
"""

import math

import torch
import torch.nn.functional as F

# United-atom radii in Angstroms: each heavy atom carries its hydrogens (ProtOr radii of
# Tsai et al. 1999, averaged per element). Other elements get the default radius.
UNITED_ATOM_RADII: dict[str, float] = {
    "C": 1.76,
    "N": 1.64,
    "O": 1.46,
    "P": 1.80,
    "S": 1.77,
}
DEFAULT_UNITED_ATOM_RADIUS = 1.7
SOLVENT_PROBE_RADIUS = 1.4  # Angstroms, a water molecule
EXCLUDED_VOLUME_EDGE_SIGMA = 0.5  # Angstroms, width of the erf edge
WATER_SOLVENT_DENSITY = 0.60  # water's potential relative to the molecule's mean
HYDROGEN_IDS = ("H",)

# E is computed where its erf edge is above ~1e-9; beyond EDGE_CUTOFF sigma it is 0 or 1
EDGE_CUTOFF = 6.0
# Number of (atom, voxel) pairs evaluated at a time when placing atoms
_PAIRS_PER_BATCH = 1 << 23


def united_atom_radii(atom_ids: list[str]) -> list[float]:
    """United-atom radii in Angstroms for a list of uppercase element symbols."""
    return [UNITED_ATOM_RADII.get(a, DEFAULT_UNITED_ATOM_RADIUS) for a in atom_ids]


def _ball_offsets(radius_vox: float, device: torch.device) -> torch.Tensor:
    """Integer offsets (K, 3) of the voxels within `radius_vox` of the origin."""
    h = math.ceil(radius_vox)
    r = torch.arange(-h, h + 1, device=device)
    off = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    return off[(off * off).sum(-1) <= radius_vox**2]


def _solvent_accessible_mask(
    voxel_positions: torch.Tensor,
    reach: torch.Tensor,
    pixel_spacing: float,
    volume_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Voxels closer than `reach` (Angstroms) to an atom centre.

    Parameters
    ----------
    voxel_positions : torch.Tensor
        Atom centres in voxel coordinates (z, y, x), shape (N, 3), float32.
    reach : torch.Tensor
        Per-atom reach in Angstroms, shape (N,), float32.
    pixel_spacing : float
        Voxel size in Angstroms.
    volume_shape : tuple[int, int, int]
        Shape of the volume.

    Returns
    -------
    torch.Tensor
        Boolean mask of the volume's shape.
    """
    device = voxel_positions.device
    mask = torch.zeros(volume_shape, dtype=torch.bool, device=device)
    if voxel_positions.shape[0] == 0:
        return mask

    # |voxel - rounded centre| <= reach + sqrt(3) / 2 voxels
    offsets = _ball_offsets(float(reach.max()) / pixel_spacing + 1.0, device)
    shape = torch.tensor(volume_shape, device=device)
    strides = torch.tensor(
        [volume_shape[1] * volume_shape[2], volume_shape[2], 1], device=device
    )
    flat = mask.view(-1)
    batch = max(1, _PAIRS_PER_BATCH // offsets.shape[0])
    for start in range(0, voxel_positions.shape[0], batch):
        pos = voxel_positions[start : start + batch]
        vox = torch.round(pos).long()[:, None, :] + offsets[None]  # (b, K, 3)
        dist = torch.linalg.vector_norm(
            (vox.to(pos.dtype) - pos[:, None, :]) * pixel_spacing, dim=-1
        )
        keep = (
            (dist < reach[start : start + batch, None])
            & (vox >= 0).all(-1)
            & (vox < shape).all(-1)
        )
        flat[(vox[keep] * strides).sum(-1)] = True

    return mask


def _squared_distance_to_mask(mask: torch.Tensor, max_offset: int) -> torch.Tensor:
    """Squared Euclidean distance (voxels^2) from each voxel to the nearest True voxel.

    Separable minimum over offsets of at most `max_offset` voxels along each axis,
    so the result is exact wherever the distance is at most `max_offset` voxels, and
    larger than `max_offset` (possibly inf) elsewhere. Voxels outside the volume do
    not count as True.
    """
    dist = torch.where(mask, 0.0, math.inf).to(torch.float32)
    for dim in range(3):
        n = dist.shape[dim]
        pad = [0, 0] * 3
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = max_offset
        padded = F.pad(dist, pad, value=math.inf)
        out = dist.clone()
        for d in range(1, max_offset + 1):
            for start in (max_offset + d, max_offset - d):
                torch.minimum(out, padded.narrow(dim, start, n) + float(d * d), out=out)
        dist = out

    return dist


def solvent_excluded_volume(
    atom_positions_zyx: torch.Tensor,
    atom_ids: list[str],
    pixel_spacing: float,
    volume_shape: tuple[int, int, int],
    probe_radius: float = SOLVENT_PROBE_RADIUS,
    edge_sigma: float = EXCLUDED_VOLUME_EDGE_SIGMA,
) -> torch.Tensor:
    """The solvent-excluded volume of an atomic model, 1 inside and 0 outside.

    Hydrogens are dropped and each heavy atom gets its united-atom radius
    (`UNITED_ATOM_RADII`). The solvent-accessible volume (radii plus the probe) is
    eroded by the probe with a Euclidean distance transform, which gives the
    solvent-excluded volume. Its edge is an erf ramp of width `edge_sigma` in the
    signed distance to that surface.

    Atoms are placed on the grid as `simulate3d` places them: an atom at the origin
    sits at voxel (D / 2, H / 2, W / 2), the box centre.

    The erosion measures distances between voxel centres, so the surface sits up to
    about half a voxel further out than the exact solvent-excluded surface: for one
    carbon atom the volume is about 3 % too large at 0.1 A per voxel and 20-40 % at
    0.5-1 A. For a large molecule the error is a shell at its surface: for a
    ribosomal subunit, the volume is about 6 % larger at 0.94 A per voxel and 15 %
    larger at 1.2 A than on finer grids. At a voxel size of `probe_radius` or more
    the erosion cannot act, so that raises an error.

    Parameters
    ----------
    atom_positions_zyx : torch.Tensor
        Atom positions in Angstroms, shape (N, 3), as passed to `simulate3d`.
    atom_ids : list[str]
        Uppercase element symbols of the atoms.
    pixel_spacing : float
        Voxel size of the volume in Angstroms.
    volume_shape : tuple[int, int, int]
        Shape of the volume.
    probe_radius : float
        Solvent probe radius in Angstroms. Default is 1.4 (water).
    edge_sigma : float
        Standard deviation of the erf edge in Angstroms. Default is 0.5.

    Returns
    -------
    torch.Tensor
        The excluded volume (float32, values in [0, 1]) on the device of
        `atom_positions_zyx`.
    """
    if pixel_spacing <= 0:
        raise ValueError(f"pixel_spacing must be positive; got {pixel_spacing}.")
    if probe_radius < 0:
        raise ValueError(f"probe_radius must be >= 0; got {probe_radius}.")
    if 0 < probe_radius <= pixel_spacing:
        raise ValueError(
            f"The voxel size ({pixel_spacing} A) must be smaller than the probe radius "
            f"({probe_radius} A): the probe erosion works on voxel centres and cannot "
            "act on a coarser grid. Simulate at a finer pixel spacing."
        )
    if edge_sigma <= 0:
        raise ValueError(f"edge_sigma must be positive; got {edge_sigma}.")
    if len(atom_ids) != atom_positions_zyx.shape[0]:
        raise ValueError(
            f"Got {atom_positions_zyx.shape[0]} atom positions but "
            f"{len(atom_ids)} atom ids."
        )

    device = atom_positions_zyx.device
    heavy = [i for i, a in enumerate(atom_ids) if a not in HYDROGEN_IDS]
    zyx = atom_positions_zyx[torch.tensor(heavy, dtype=torch.long, device=device)]
    centre = torch.tensor([s / 2 for s in volume_shape], dtype=torch.float64)
    voxel_positions = (zyx.double() / pixel_spacing + centre.to(device)).float()
    reach = torch.tensor(
        [r + probe_radius for r in united_atom_radii([atom_ids[i] for i in heavy])],
        dtype=torch.float32,
        device=device,
    )

    excluded = torch.zeros(volume_shape, dtype=torch.float32, device=device)
    sas = _solvent_accessible_mask(voxel_positions, reach, pixel_spacing, volume_shape)
    if not bool(sas.any()):
        return excluded

    # Work in the bounding box of the accessible volume plus a margin that holds every
    # offset the distance transforms look at; beyond it the erf edge is below ~1e-9.
    erode = math.ceil(probe_radius / pixel_spacing)
    edge_reach = math.ceil(EDGE_CUTOFF * edge_sigma / pixel_spacing + 0.5)
    margin = max(erode, edge_reach) + 1
    box = []
    for dim in range(3):
        other = tuple(d for d in range(3) if d != dim)
        hit = torch.nonzero(sas.any(dim=other)).flatten()
        lo = max(int(hit[0]) - margin, 0)
        hi = min(int(hit[-1]) + margin + 1, volume_shape[dim])
        box.append(slice(lo, hi))
    sas = sas[box[0], box[1], box[2]]

    # Erode the accessible volume by the probe: keep voxels at least `probe_radius`
    # from the nearest inaccessible voxel.
    if erode > 0:
        d2 = _squared_distance_to_mask(~sas, erode)
        ses = sas & (torch.sqrt(d2.double()) * pixel_spacing >= probe_radius)
        del d2
    else:
        ses = sas
    del sas

    # Signed distance to the surface (negative inside), halfway between voxel centres
    d_in = torch.sqrt(_squared_distance_to_mask(~ses, edge_reach)) * pixel_spacing
    d_out = torch.sqrt(_squared_distance_to_mask(ses, edge_reach)) * pixel_spacing
    signed = torch.where(ses, 0.5 * pixel_spacing - d_in, d_out - 0.5 * pixel_spacing)
    del d_in, d_out
    excluded[box[0], box[1], box[2]] = 0.5 * torch.special.erfc(
        signed / (math.sqrt(2.0) * edge_sigma)
    )

    return excluded


def mean_inside_excluded_volume(
    potential: torch.Tensor, excluded_volume: torch.Tensor
) -> float:
    """Mean of `potential` over the voxels where `excluded_volume` >= 0.5.

    This is the scale that turns the unit excluded volume into the molecule's mean
    potential, so that the solvent density is a fraction of it.
    """
    inside = excluded_volume >= 0.5
    count = int(inside.sum())
    if count == 0:
        raise ValueError(
            "The excluded volume has no voxel inside (>= 0.5), so it cannot be "
            "scaled to the molecule's potential. Is the model empty, or only "
            "hydrogens?"
        )
    return float(potential[inside].double().sum()) / count
