"""The solvent a molecule displaces: its solvent-excluded volume on the simulation grid.

ttsim3d simulates a molecule in vacuum. In a specimen the molecule displaces solvent,
whose own potential is missing from the vacuum map. Subtracting it gives the template

    A - rho * E,

where A is the simulated potential before any dose or MTF filter, E is the molecule's
solvent-excluded volume scaled to the mean of A inside it, and rho is the solvent's
potential as a fraction of that mean (about 0.6 for water against protein and RNA).

E is computed from the exact geometry of the atoms, not from voxel centres, so it does
not depend on the pixel size: see `solvent_excluded_volume`.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch

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
SURFACE_SEED_SPACING = 0.5  # Angstroms, largest spacing of the surface seed grid
HYDROGEN_IDS = ("H",)

# E is computed where its erf edge is above ~1e-9; beyond EDGE_CUTOFF sigma it is 0 or 1
EDGE_CUTOFF = 6.0
# Seeds are seed-grid points outside the accessible volume and within this many seed
# voxels of it; their surface points are kept within _SEED_REACH seed bands of them.
_SEED_BAND = math.sqrt(3.0)
_SEED_REACH = 2.0
_INSIDE_TOLERANCE = 1e-3  # Angstroms: a point this far inside an atom is still outside
_PAIRS_PER_BATCH = 1 << 23
_POINTS_PER_BATCH = 1 << 22


def united_atom_radii(atom_ids: list[str]) -> list[float]:
    """United-atom radii in Angstroms for a list of uppercase element symbols."""
    return [UNITED_ATOM_RADII.get(a, DEFAULT_UNITED_ATOM_RADIUS) for a in atom_ids]


def _ball_offsets(radius: float, device: torch.device) -> torch.Tensor:
    """Integer offsets (K, 3) of the voxels within `radius` voxels of the origin."""
    h = math.ceil(radius)
    r = torch.arange(-h, h + 1, device=device)
    off = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    return off[(off * off).sum(-1) <= radius**2]


def _neighbourhoods(
    points: torch.Tensor, radius: float, shape: tuple[int, int, int]
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Voxels within `radius` (voxels) of each point, in chunks of points.

    Yields (start, points, voxels (b, K, 3), flat voxel indices (b, K), in bounds).
    """
    device = points.device
    offsets = _ball_offsets(radius, device)
    bounds = torch.tensor(shape, device=device)
    strides = torch.tensor([shape[1] * shape[2], shape[2], 1], device=device)
    batch = max(1, _PAIRS_PER_BATCH // offsets.shape[0])
    for start in range(0, points.shape[0], batch):
        p = points[start : start + batch]
        vox = torch.round(p).long()[:, None, :] + offsets[None]
        inside = (vox >= 0).all(-1) & (vox < bounds).all(-1)
        flat = torch.where(inside, (vox * strides).sum(-1), 0)
        yield start, p, vox, flat, inside


def _depth(
    centres: torch.Tensor,
    radii: torch.Tensor,
    spacing: float,
    shape: tuple[int, int, int],
    reach: float,
) -> torch.Tensor:
    """max_i (R_i - |x - c_i|) in Angstroms, flattened; -inf below -reach.

    `centres` are in voxels of size `spacing`. Positive inside the union of the balls,
    where it is a lower bound on the distance to its surface; outside, it is minus the
    exact distance to the union.
    """
    depth = torch.full(
        (shape[0] * shape[1] * shape[2],), -math.inf, device=centres.device
    )
    radius = (float(radii.max()) + reach) / spacing + 1
    for start, p, vox, flat, inside in _neighbourhoods(centres, radius, shape):
        value = radii[start : start + p.shape[0], None] - torch.linalg.vector_norm(
            (vox.float() - p[:, None]) * spacing, dim=-1
        )
        keep = inside & (value > -reach)
        depth.scatter_reduce_(0, flat[keep], value[keep], "amax")
    return depth


def _unravel(flat: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Flat indices to (N, 3) integer coordinates."""
    return torch.stack(
        [flat // (shape[1] * shape[2]), (flat // shape[2]) % shape[1], flat % shape[2]],
        dim=-1,
    )


def _normalise(v: torch.Tensor) -> torch.Tensor:
    unit: torch.Tensor = v / torch.linalg.vector_norm(
        v, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    return unit


def _circle_points(
    u: torch.Tensor,
    ci: torch.Tensor,
    ri: torch.Tensor,
    cj: torch.Tensor,
    rj: torch.Tensor,
) -> torch.Tensor:
    """Point nearest to u on the intersection circle of two spheres; nan if none."""
    d = torch.linalg.vector_norm(cj - ci, dim=-1).clamp_min(1e-12)
    n = (cj - ci) / d[:, None]
    t = (d * d + ri * ri - rj * rj) / (2 * d)
    rho2 = ri * ri - t * t
    w = u - ci
    radial = w - (w * n).sum(-1, keepdim=True) * n
    ok = (rho2 > 0) & (torch.linalg.vector_norm(radial, dim=-1) > 1e-6)
    q = (
        ci
        + t[:, None] * n
        + torch.sqrt(rho2.clamp_min(0))[:, None] * _normalise(radial)
    )
    return torch.where(ok[:, None], q, math.nan)


def _corner_points(
    u: torch.Tensor,
    ci: torch.Tensor,
    ri: torch.Tensor,
    cj: torch.Tensor,
    rj: torch.Tensor,
    ck: torch.Tensor,
    rk: torch.Tensor,
) -> torch.Tensor:
    """Intersection point of three spheres nearest to u; nan if none."""
    d = torch.linalg.vector_norm(cj - ci, dim=-1).clamp_min(1e-12)
    ex = (cj - ci) / d[:, None]
    a = ((ck - ci) * ex).sum(-1)
    ey_raw = ck - ci - a[:, None] * ex
    b = torch.linalg.vector_norm(ey_raw, dim=-1).clamp_min(1e-12)
    ey = ey_raw / b[:, None]
    ez = torch.linalg.cross(ex, ey, dim=-1)
    x = (ri * ri - rj * rj + d * d) / (2 * d)
    y = (ri * ri - rk * rk + a * a + b * b) / (2 * b) - (a / b) * x
    z2 = ri * ri - x * x - y * y
    ok = (z2 > 0) & (b > 1e-6)
    base = ci + x[:, None] * ex + y[:, None] * ey
    z = torch.sqrt(z2.clamp_min(0))[:, None] * ez
    first = torch.linalg.vector_norm(base + z - u, dim=-1) <= torch.linalg.vector_norm(
        base - z - u, dim=-1
    )
    q = torch.where(first[:, None], base + z, base - z)
    return torch.where(ok[:, None], q, math.nan)


def _outside_atoms(
    points: torch.Tensor,
    atoms: torch.Tensor,
    centres: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """True where a point (N, 3) is in none of its atoms (N, K; -1 is padding)."""
    index = atoms.clamp_min(0)
    reach = torch.where(atoms >= 0, radii[index], -math.inf) - _INSIDE_TOLERANCE
    distance = torch.linalg.vector_norm(points[:, None, :] - centres[index], dim=-1)
    outside: torch.Tensor = ~(distance < reach).any(-1)
    return outside


@dataclass
class _SurfacePoints:
    """Points on the accessible surface, with the feature each one lies on.

    kind is 0 for a sphere (atom a), 1 for the circle of atoms a and b, 2 for a
    corner. seed is the seed each point came from; every atom that can contain a point
    within `reach` of a seed is in its row of seed_atoms.
    """

    points: torch.Tensor  # (M, 3), Angstroms
    kind: torch.Tensor
    atom_a: torch.Tensor
    atom_b: torch.Tensor
    seed: torch.Tensor
    seed_points: torch.Tensor  # (S, 3), Angstroms
    seed_atoms: torch.Tensor  # (S, K)
    reach: float


def _seed_atoms(
    centres: torch.Tensor,
    radii: torch.Tensor,
    spacing: float,
    shape: tuple[int, int, int],
    seeds: torch.Tensor,
    reach: float,
) -> torch.Tensor:
    """Every atom that can contain a point within `reach` of each seed, nearest first.

    Returns (S, K) atom indices, padded with -1.
    """
    device = centres.device
    n_seeds = seeds.numel()
    seed_of, atom_of, value_of = [], [], []
    radius = (float(radii.max()) + reach) / spacing + 1
    for start, p, vox, flat, inside in _neighbourhoods(centres, radius, shape):
        value = radii[start : start + p.shape[0], None] - torch.linalg.vector_norm(
            (vox.float() - p[:, None]) * spacing, dim=-1
        )
        keep = inside & (value >= -reach)
        index = torch.searchsorted(seeds, flat[keep]).clamp_max(n_seeds - 1)
        hit = seeds[index] == flat[keep]
        atom = torch.arange(start, start + p.shape[0], device=device)
        seed_of.append(index[hit])
        atom_of.append(atom[:, None].expand_as(value)[keep][hit])
        value_of.append(value[keep][hit])
    seed_index, atom_index = torch.cat(seed_of), torch.cat(atom_of)
    order = torch.argsort(torch.cat(value_of), descending=True, stable=True)
    seed_index, atom_index = seed_index[order], atom_index[order]
    order = torch.argsort(seed_index, stable=True)
    seed_index, atom_index = seed_index[order], atom_index[order]
    counts = torch.bincount(seed_index, minlength=n_seeds)
    first = counts.cumsum(0) - counts
    rank = torch.arange(seed_index.numel(), device=device) - first[seed_index]
    width = max(int(counts.max()), 3)
    atoms = torch.full((n_seeds, width), -1, dtype=torch.long, device=device)
    atoms[seed_index, rank] = atom_index
    return atoms


def _surface_points(
    centres: torch.Tensor,
    radii: torch.Tensor,
    spacing: float,
    shape: tuple[int, int, int],
) -> _SurfacePoints:
    """Points on the accessible surface, from seeds on a grid of `spacing`.

    Each seed (a grid point just outside the accessible volume) gives the nearest point
    of its nearest sphere, the nearest points of the circles of its three nearest
    spheres, and their corner. All of them lie on the surface: the sphere point by
    construction, the others once checked against every atom near the seed.
    """
    device = centres.device
    band = _SEED_BAND * spacing
    reach = _SEED_REACH * band
    centres_a = centres * spacing
    depth = _depth(centres, radii, spacing, shape, reach)
    seeds = torch.nonzero((depth <= 0) & (depth >= -band)).flatten()
    del depth
    seed_points = _unravel(seeds, shape).float() * spacing
    if seeds.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return _SurfacePoints(
            points=seed_points,
            kind=empty,
            atom_a=empty,
            atom_b=empty,
            seed=empty,
            seed_points=seed_points,
            seed_atoms=torch.zeros((0, 3), dtype=torch.long, device=device),
            reach=reach,
        )
    seed_atoms = _seed_atoms(centres, radii, spacing, shape, seeds, reach)

    points, kinds, atoms_a, atoms_b, seed_ids = [], [], [], [], []

    def add(
        q: torch.Tensor,
        keep: torch.Tensor,
        kind: int,
        a: torch.Tensor,
        b: torch.Tensor,
        ids: torch.Tensor,
    ) -> None:
        points.append(q[keep])
        kinds.append(torch.full_like(a[keep], kind))
        atoms_a.append(a[keep])
        atoms_b.append(b[keep])
        seed_ids.append(ids[keep])

    batch = max(1, _POINTS_PER_BATCH // seed_atoms.shape[1])
    for start in range(0, seeds.numel(), batch):
        u = seed_points[start : start + batch]
        atoms = seed_atoms[start : start + batch]
        c = centres_a[atoms.clamp_min(0)]
        r = torch.where(atoms >= 0, radii[atoms.clamp_min(0)], math.nan)
        ids = torch.arange(start, start + u.shape[0], device=device)
        none = torch.full_like(atoms[:, 0], -1)

        q = c[:, 0] + _normalise(u - c[:, 0]) * r[:, 0, None]
        add(q, torch.isfinite(q).all(-1), 0, atoms[:, 0], none, ids)
        candidates = [
            (
                1,
                atoms[:, i],
                atoms[:, j],
                _circle_points(u, c[:, i], r[:, i], c[:, j], r[:, j]),
            )
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
        corner = _corner_points(u, c[:, 0], r[:, 0], c[:, 1], r[:, 1], c[:, 2], r[:, 2])
        candidates.append((2, atoms[:, 0], none, corner))
        for kind, a, b, q in candidates:
            near = torch.linalg.vector_norm(q - u, dim=-1) <= reach
            keep = (
                torch.isfinite(q).all(-1)
                & near
                & _outside_atoms(q, atoms, centres_a, radii)
            )
            add(q, keep, kind, a, b, ids)

    return _SurfacePoints(
        points=torch.cat(points),
        kind=torch.cat(kinds),
        atom_a=torch.cat(atoms_a),
        atom_b=torch.cat(atoms_b),
        seed=torch.cat(seed_ids),
        seed_points=seed_points,
        seed_atoms=seed_atoms,
        reach=reach,
    )


def _jump_flood(
    points: torch.Tensor, shape: tuple[int, int, int], max_distance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest point (index, distance in voxels) for every voxel within max_distance.

    `points` are in voxels. Voxels next to a point start with it; jump flooding then
    passes each voxel's point to its neighbours at halving steps. Voxels farther than
    max_distance get -1 and inf.
    """
    device = points.device
    n = shape[0] * shape[1] * shape[2]
    distance = torch.full((n,), math.inf, device=device)
    nearest = torch.full((n,), -1, dtype=torch.long, device=device)
    if points.shape[0] == 0:
        return nearest, distance
    for _, p, vox, flat, inside in _neighbourhoods(points, 1.0, shape):
        d = torch.linalg.vector_norm(vox.float() - p[:, None], dim=-1)
        distance.scatter_reduce_(0, flat[inside], d[inside], "amin")
    for start, p, vox, flat, inside in _neighbourhoods(points, 1.0, shape):
        d = torch.linalg.vector_norm(vox.float() - p[:, None], dim=-1)
        keep = inside & (d == distance[flat])
        index = torch.arange(start, start + p.shape[0], device=device)
        nearest[flat[keep]] = index[:, None].expand_as(d)[keep]

    nearest3, distance3 = nearest.view(shape), distance.view(shape)
    axes = [torch.arange(m, device=device, dtype=torch.float32) for m in shape]
    steps = []
    step = 1 << max(0, math.floor(math.log2(max(max_distance, 1.0))))
    while step >= 1:
        steps.append(step)
        step //= 2
    steps.append(1)
    offsets = [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ]
    for step in steps:
        for offset in offsets:
            dst, src = [], []
            for axis, o in enumerate(offset):
                o *= step
                dst.append(slice(max(0, -o), shape[axis] - max(0, o)))
                src.append(slice(max(0, o), shape[axis] - max(0, -o)))
            if any(s.stop <= s.start for s in dst):
                continue
            candidate = nearest3[src[0], src[1], src[2]]
            p = points[candidate.clamp_min(0)]
            d = torch.sqrt(
                (axes[0][dst[0]][:, None, None] - p[..., 0]) ** 2
                + (axes[1][dst[1]][None, :, None] - p[..., 1]) ** 2
                + (axes[2][dst[2]][None, None, :] - p[..., 2]) ** 2
            )
            current = distance3[dst[0], dst[1], dst[2]]
            better = (candidate >= 0) & (d < current) & (d <= max_distance)
            distance3[dst[0], dst[1], dst[2]] = torch.where(better, d, current)
            nearest3[dst[0], dst[1], dst[2]] = torch.where(
                better, candidate, nearest3[dst[0], dst[1], dst[2]]
            )
    return nearest, distance


def _snap_to_features(
    x: torch.Tensor,
    surface: _SurfacePoints,
    nearest: torch.Tensor,
    centres: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Distance from x to the point of its nearest point's feature nearest to x.

    That point is the projection of x onto the sphere, the nearest point of the circle,
    or the corner itself. Where it is not on the surface (inside an atom near its seed,
    or out of the seed's reach) the distance is inf.
    """
    kind = surface.kind[nearest]
    a = surface.atom_a[nearest]
    b = surface.atom_b[nearest].clamp_min(0)
    ca, ra, cb, rb = centres[a], radii[a], centres[b], radii[b]
    on_sphere = ca + _normalise(x - ca) * ra[:, None]
    on_circle = _circle_points(x, ca, ra, cb, rb)
    q = torch.where(
        (kind == 0)[:, None],
        on_sphere,
        torch.where((kind == 1)[:, None], on_circle, surface.points[nearest]),
    )
    seed = surface.seed[nearest]
    ok = torch.isfinite(q).all(-1) & (
        torch.linalg.vector_norm(q - surface.seed_points[seed], dim=-1) <= surface.reach
    )
    for start in range(0, q.shape[0], 1 << 16):
        part = slice(start, start + (1 << 16))
        ok[part] &= _outside_atoms(
            q[part], surface.seed_atoms[seed[part]], centres, radii
        )
    distance = torch.linalg.vector_norm(x - q, dim=-1)
    return torch.where(ok, distance, math.inf)


def solvent_excluded_volume(
    atom_positions_zyx: torch.Tensor,
    atom_ids: list[str],
    pixel_spacing: float,
    volume_shape: tuple[int, int, int],
    probe_radius: float = SOLVENT_PROBE_RADIUS,
    edge_sigma: float = EXCLUDED_VOLUME_EDGE_SIGMA,
    seed_spacing: float = SURFACE_SEED_SPACING,
) -> torch.Tensor:
    """The solvent-excluded volume of an atomic model, 1 inside and 0 outside.

    Hydrogens are dropped and each heavy atom gets its united-atom radius
    (`UNITED_ATOM_RADII`). The solvent-accessible volume is the union of the atoms'
    balls with radii plus the probe; the probe's centre can reach everything outside
    it. The solvent-excluded volume holds the points at least `probe_radius` from that
    region, so the signed distance to its surface is `probe_radius - D`, where D is the
    distance to the accessible surface (and minus the distance to the accessible
    volume outside it). E is an erf ramp of width `edge_sigma` in that signed distance.

    D is computed from the exact geometry, not from voxel centres, so E does not depend
    on the pixel size:
    - points on the accessible surface are found from seeds on an internal grid of at
      most `seed_spacing`: each seed gives the nearest point of its nearest sphere,
      and the nearest points of the circles and the corner where its nearest spheres
      meet, each checked against every atom near the seed;
    - each voxel takes its nearest such point (jump flooding), and then the exact
      distance to the sphere, circle or corner that point lies on, where its nearest
      point there is checked to be on the surface too.
    One atom gives the analytic erf-edged sphere; for fragments of a ribosomal subunit
    the volume is within about 2 % of an exact reference at pixel sizes of 0.5-2 A.
    Gaps between atoms narrower than about the seed spacing can be missed. Enclosed
    cavities that hold a seed count as solvent.

    Atoms are placed on the grid as `simulate3d` places them: an atom at the origin
    sits at voxel (D / 2, H / 2, W / 2), the box centre.

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
    seed_spacing : float
        Largest spacing in Angstroms of the grid that seeds the surface points; the
        grid divides the voxels evenly. Smaller is more accurate and slower. Default
        is 0.5.

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
    if edge_sigma <= 0:
        raise ValueError(f"edge_sigma must be positive; got {edge_sigma}.")
    if seed_spacing <= 0:
        raise ValueError(f"seed_spacing must be positive; got {seed_spacing}.")
    if len(atom_ids) != atom_positions_zyx.shape[0]:
        raise ValueError(
            f"Got {atom_positions_zyx.shape[0]} atom positions but "
            f"{len(atom_ids)} atom ids."
        )

    device = atom_positions_zyx.device
    excluded = torch.zeros(volume_shape, dtype=torch.float32, device=device)
    heavy = [i for i, a in enumerate(atom_ids) if a not in HYDROGEN_IDS]
    if not heavy:
        return excluded

    zyx = atom_positions_zyx[torch.tensor(heavy, dtype=torch.long, device=device)]
    box_centre = torch.tensor([s / 2 for s in volume_shape], dtype=torch.float64)
    positions = zyx.double() / pixel_spacing + box_centre.to(device)
    radii = torch.tensor(
        [r + probe_radius for r in united_atom_radii([atom_ids[i] for i in heavy])],
        dtype=torch.float32,
        device=device,
    )

    edge = EDGE_CUTOFF * edge_sigma
    subdivision = max(1, math.ceil(pixel_spacing / seed_spacing - 1e-9))
    seed_grid = pixel_spacing / subdivision
    outside_reach = max(edge - probe_radius, 0.0)
    seed_reach = _SEED_REACH * _SEED_BAND * seed_grid

    # work on the bounding box of everything the atoms reach, in voxels of the volume
    margin = (
        math.ceil((float(radii.max()) + max(outside_reach, seed_reach)) / pixel_spacing)
        + 2
    )
    low = torch.floor(positions.min(0).values).long() - margin
    high = torch.ceil(positions.max(0).values).long() + margin + 1
    shape = (int(high[0] - low[0]), int(high[1] - low[1]), int(high[2] - low[2]))
    seed_shape = (
        (shape[0] - 1) * subdivision + 1,
        (shape[1] - 1) * subdivision + 1,
        (shape[2] - 1) * subdivision + 1,
    )
    centres = (positions - low).float()  # voxels of the work box
    centres_a = centres * pixel_spacing

    depth = _depth(centres, radii, pixel_spacing, shape, outside_reach)
    surface = _surface_points(centres * subdivision, radii, seed_grid, seed_shape)
    max_distance = probe_radius + edge + seed_grid
    nearest, distance = _jump_flood(
        surface.points / pixel_spacing, shape, max_distance / pixel_spacing
    )
    distance *= pixel_spacing

    voxels = torch.nonzero((nearest >= 0) & (depth > 0)).flatten()
    x = _unravel(voxels, shape).float() * pixel_spacing
    snapped = _snap_to_features(x, surface, nearest[voxels], centres_a, radii)
    distance[voxels] = torch.minimum(distance[voxels], snapped)
    del surface, nearest

    # distance to the accessible surface inside, minus the distance to it outside
    signed = probe_radius - torch.where(depth > 0, distance, depth)
    values = 0.5 * torch.special.erfc(signed / (math.sqrt(2.0) * edge_sigma))
    values = values.reshape(shape)

    dst, src = [], []
    for axis in range(3):
        start, stop = int(low[axis]), int(low[axis]) + shape[axis]
        first, last = max(start, 0), min(stop, volume_shape[axis])
        if last <= first:
            return excluded
        dst.append(slice(first, last))
        src.append(slice(first - start, last - start))
    excluded[dst[0], dst[1], dst[2]] = values[src[0], src[1], src[2]]
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
