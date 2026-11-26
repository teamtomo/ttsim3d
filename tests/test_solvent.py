"""Tests for subtracting the displaced solvent (ttsim3d.solvent and simulate3d)."""

import math
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
from click.testing import CliRunner
from pydantic import ValidationError
from scipy.special import erfc
from src.programs.run_ttsim3d import run_simulation_cli
from torch_fourier_filter.mtf import read_mtf

from ttsim3d.models import DEFAULT_MTF_REFERENCES, Simulator, SimulatorConfig
from ttsim3d.simulate3d import apply_simulation_filters, simulate3d
from ttsim3d.solvent import (
    UNITED_ATOM_RADII,
    WATER_SOLVENT_DENSITY,
    mean_inside_excluded_volume,
    solvent_excluded_volume,
)

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
ELEMENTS = ["C", "N", "O", "C", "S", "P", "H", "MG", "C", "N", "O"]
PIXEL = 0.8  # auto upsampling is 2 here, so the upsampled path is tested
SHAPE = (32, 32, 32)
PROBE, SIGMA = 1.4, 0.5


@pytest.fixture
def single_thread():
    """Run on one CPU thread: multi-threaded CPU accumulation is not bitwise stable."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def small_model(n_atoms: int = 33, seed: int = 0):
    """A compact random model (Angstroms) with hydrogens and a default-radius ion."""
    rng = np.random.default_rng(seed)
    zyx = torch.tensor(rng.uniform(-4.5, 4.5, size=(n_atoms, 3)), dtype=torch.float32)
    ids = [ELEMENTS[i % len(ELEMENTS)] for i in range(n_atoms)]
    b_factors = torch.full((n_atoms,), 20.0)
    return zyx, ids, b_factors


def write_pdb(path: Path, zyx: torch.Tensor, ids: list[str]) -> None:
    """Write atoms (z, y, x in Angstroms) as a minimal PDB file."""
    lines = []
    for i, ((z, y, x), el) in enumerate(zip(zyx.tolist(), ids)):
        name = f" {el:<3s}" if len(el) == 1 else f"{el:<4s}"
        lines.append(
            f"ATOM  {i + 1:5d} {name} ALA A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {el:>2s}"
        )
    path.write_text("\n".join([*lines, "END", ""]))


def mtf_kwargs(device: str) -> dict:
    frequencies, amplitudes = read_mtf(file_path=DEFAULT_MTF_REFERENCES["k2_300kv"])
    return {
        "apply_dqe": True,
        "mtf_frequencies": frequencies.to(device),
        "mtf_amplitudes": amplitudes.to(device),
    }


def simulation_kwargs(device: str, dose: bool = True, mtf: bool = True) -> dict:
    zyx, ids, b_factors = small_model()
    kwargs = {
        "atom_positions_zyx": zyx,
        "atom_ids": ids,
        "atom_b_factors": b_factors,
        "beam_energy_kev": 300.0,
        "sim_pixel_spacing": PIXEL,
        "sim_volume_shape": SHAPE,
        "apply_dose_weighting": dose,
        "dose_start": 0.0,
        "dose_end": 50.0,
        "dose_filter_modify_signal": "rel_diff",
        "device": device,
    }
    if mtf:
        kwargs.update(mtf_kwargs(device))
    return kwargs


def grid(pixel: float, shape: tuple[int, int, int]) -> np.ndarray:
    """Voxel centres (Angstroms) as simulate3d places atoms: the origin at shape / 2."""
    axes = [(np.arange(s) - s / 2) * pixel for s in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


def exact_excluded_volume(centres, radii, pixel, shape):
    """Exact E for one or two atoms.

    Inside the accessible volume the distance to its surface is the distance to the
    exposed part of either sphere: the radial projection if it is exposed, else the
    circle where the spheres meet. Outside, it is the distance to the nearer ball.
    """
    x = grid(pixel, shape)
    reach = [r + PROBE for r in radii]
    c = [np.asarray(ci, float) for ci in centres]
    if len(c) == 1:  # a second atom far away changes nothing
        c.append(c[0] + 1e3)
        reach.append(reach[0])
    rel = [x - ci for ci in c]
    norm = [np.linalg.norm(v, axis=-1) for v in rel]
    inside = (norm[0] < reach[0]) | (norm[1] < reach[1])

    axis = c[1] - c[0]
    d = np.linalg.norm(axis)
    axis = axis / d
    t = (d * d + reach[0] ** 2 - reach[1] ** 2) / (2 * d)
    rho = math.sqrt(max(reach[0] ** 2 - t * t, 0.0))
    along = rel[0] @ axis
    across = np.linalg.norm(rel[0] - along[..., None] * axis, axis=-1)
    to_circle = np.sqrt((along - t) ** 2 + (across - rho) ** 2)

    to_surface = []
    for i, j in ((0, 1), (1, 0)):
        q = c[i] + rel[i] * (reach[i] / np.maximum(norm[i], 1e-12))[..., None]
        exposed = np.linalg.norm(q - c[j], axis=-1) >= reach[j]
        to_surface.append(np.where(exposed, np.abs(reach[i] - norm[i]), to_circle))
    depth = np.where(
        inside,
        np.minimum(*to_surface),
        -np.minimum(norm[0] - reach[0], norm[1] - reach[1]),
    )
    return 0.5 * erfc((PROBE - depth) / (math.sqrt(2) * SIGMA))


def centroid(volume: torch.Tensor) -> np.ndarray:
    v = volume.double().cpu()
    out = []
    for axis in range(3):
        profile = v.sum(dim=tuple(d for d in range(3) if d != axis))
        out.append(float((profile * torch.arange(v.shape[axis])).sum() / v.sum()))
    return np.array(out)


# --- the excluded volume itself ---------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pixel", [0.3, 0.936, 1.5, 2.0])
def test_one_atom_is_an_erf_sphere(device, pixel):
    """E of one carbon is the analytic erf-edged sphere, at any voxel size."""
    n = int(math.ceil(14 / pixel)) // 2 * 2
    atom = [0.37, -0.21, 0.13]
    excluded = solvent_excluded_volume(
        torch.tensor([atom], device=device), ["C"], pixel, (n, n, n)
    )
    assert excluded.dtype == torch.float32
    assert excluded.device.type == torch.device(device).type
    exact = exact_excluded_volume([atom], [UNITED_ATOM_RADII["C"]], pixel, (n, n, n))
    np.testing.assert_allclose(excluded.cpu().numpy(), exact, rtol=0, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pixel", [0.5, 0.936, 1.5])
@pytest.mark.parametrize("distance", [1.45, 3.5, 5.5])
def test_two_atoms_match_the_exact_surface(device, pixel, distance):
    """Two atoms: the crease where the probe touches both is placed exactly too."""
    n = int(math.ceil(18 / pixel)) // 2 * 2
    rng = np.random.default_rng(int(distance * 10))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    offset = rng.uniform(-0.5, 0.5, 3) * pixel
    atoms = np.stack(
        [offset - direction * distance / 2, offset + direction * distance / 2]
    )
    excluded = solvent_excluded_volume(
        torch.tensor(atoms, dtype=torch.float32, device=device),
        ["C", "O"],
        pixel,
        (n, n, n),
    )
    ours = excluded.cpu().double().numpy()
    radii = [UNITED_ATOM_RADII["C"], UNITED_ATOM_RADII["O"]]
    exact = exact_excluded_volume(atoms, radii, pixel, (n, n, n))
    assert ours.sum() == pytest.approx(exact.sum(), rel=0.005)
    assert np.linalg.norm(ours - exact) / np.linalg.norm(exact) < 0.02
    assert np.abs(ours - exact).max() < 0.1


def test_volume_does_not_depend_on_pixel_size():
    zyx, ids, _ = small_model()
    volumes = []
    for pixel in (0.4, 0.8, 1.2, 1.6):
        n = int(math.ceil(28 / pixel))
        excluded = solvent_excluded_volume(zyx, ids, pixel, (n, n, n))
        volumes.append(float(excluded.double().sum()) * pixel**3)
    assert max(volumes) / min(volumes) < 1.01


def test_radii_and_hydrogens():
    pixel, shape = 0.5, (24, 24, 24)
    zyx = torch.tensor([[0.3, -0.4, 0.2]])

    def one(atom_id):
        return solvent_excluded_volume(zyx, [atom_id], pixel, shape)

    assert torch.equal(one("MG"), one("XX"))  # default radius
    assert float(one("P").sum()) > float(one("C").sum()) > float(one("O").sum())
    assert float(one("H").sum()) == 0.0

    model, ids, _ = small_model()
    without_h = [i for i, a in enumerate(ids) if a != "H"]
    assert torch.equal(
        solvent_excluded_volume(model, ids, pixel, shape),
        solvent_excluded_volume(
            model[without_h], [ids[i] for i in without_h], pixel, shape
        ),
    )


def test_excluded_volume_input_checks():
    zyx, ids, _ = small_model()
    with pytest.raises(ValueError, match="atom ids"):
        solvent_excluded_volume(zyx, ids[:-1], PIXEL, SHAPE)
    with pytest.raises(ValueError, match="seed_spacing"):
        solvent_excluded_volume(zyx, ids, PIXEL, SHAPE, seed_spacing=0.0)
    with pytest.raises(ValueError, match="no voxel inside"):
        mean_inside_excluded_volume(
            torch.ones(SHAPE), solvent_excluded_volume(zyx[:0], [], PIXEL, SHAPE)
        )


# --- subtraction in simulate3d ------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("upsampling", [1, 2])
@pytest.mark.parametrize("size", [32, 33])
def test_excluded_volume_is_placed_like_the_atoms(device, upsampling, size):
    atom = torch.tensor([[2.3, -1.7, 0.9]])
    volume, excluded = simulate3d(
        atom,
        ["S"],
        torch.tensor([10.0]),
        300.0,
        0.6,
        (size, size, size),
        requested_upsampling=upsampling,
        apply_dose_weighting=False,
        device=device,
        return_excluded_volume=True,
    )
    np.testing.assert_allclose(centroid(excluded), centroid(volume), atol=0.1)
    np.testing.assert_allclose(
        centroid(excluded), atom[0].numpy() / 0.6 + size / 2, atol=0.1
    )


@pytest.mark.parametrize("device", DEVICES)
def test_zero_solvent_density_is_unchanged(device, single_thread):
    kwargs = simulation_kwargs(device)
    today = simulate3d(**kwargs)
    assert torch.equal(simulate3d(**kwargs, solvent_density=0.0), today)
    volume, excluded = simulate3d(**kwargs, return_excluded_volume=True)
    assert torch.equal(volume, today)
    assert excluded.shape == today.shape


@pytest.mark.parametrize("device", DEVICES)
def test_excluded_volume_recombines_for_any_density(device):
    kwargs = simulation_kwargs(device)
    today, excluded = simulate3d(**kwargs, return_excluded_volume=True)
    scale = float(today.abs().max())
    for rho in (0.3, WATER_SOLVENT_DENSITY, 0.72):
        template = simulate3d(**kwargs, solvent_density=rho)
        torch.testing.assert_close(
            template, today - rho * excluded, rtol=0, atol=1e-6 * scale
        )
        with_e, e_again = simulate3d(
            **kwargs, solvent_density=rho, return_excluded_volume=True
        )
        torch.testing.assert_close(with_e, template, rtol=0, atol=1e-6 * scale)
        torch.testing.assert_close(e_again, excluded, rtol=0, atol=1e-6 * scale)


@pytest.mark.parametrize("device", DEVICES)
def test_filters_act_on_potential_minus_solvent(device):
    """The template is the dose-filtered and MTF-filtered A - rho * E."""
    rho = 0.65
    zyx, ids, _ = small_model()
    potential, unfiltered = simulate3d(
        **simulation_kwargs(device, dose=False, mtf=False),
        return_excluded_volume=True,
    )
    unit = solvent_excluded_volume(zyx.to(device), ids, PIXEL, SHAPE)
    scale = mean_inside_excluded_volume(potential, unit)
    assert scale > 0
    # E without filters only goes through an FFT round trip
    torch.testing.assert_close(unfiltered, scale * unit, rtol=0, atol=1e-6 * scale)

    template, excluded = simulate3d(
        **simulation_kwargs(device), solvent_density=rho, return_excluded_volume=True
    )
    filter_kwargs = {
        "actual_upsampling": 1,
        "upsampled_pixel_size": PIXEL,
        "final_shape": SHAPE,
        "apply_dose_weighting": True,
        "dose_start": 0.0,
        "dose_end": 50.0,
        "dose_filter_modify_signal": "rel_diff",
        "dose_filter_critical_bfactor": -1,
        **mtf_kwargs(device),
    }
    by_hand = apply_simulation_filters(potential - rho * scale * unit, **filter_kwargs)
    assert float((template - by_hand).norm() / by_hand.norm()) < 1e-5
    torch.testing.assert_close(
        excluded,
        apply_simulation_filters(scale * unit, **filter_kwargs),
        rtol=0,
        atol=1e-6 * scale,
    )


def test_negative_solvent_density_is_rejected():
    with pytest.raises(ValueError, match="solvent_density"):
        simulate3d(**simulation_kwargs("cpu"), solvent_density=-0.1)


# --- Simulator, SimulatorConfig and the CLI -----------------------------------------


def test_simulator_config_solvent_fields():
    config = SimulatorConfig()
    assert config.subtract_solvent is False
    assert config.solvent_density == WATER_SOLVENT_DENSITY == 0.6
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            SimulatorConfig(subtract_solvent=True, solvent_density=bad)


@pytest.mark.parametrize("device", DEVICES)
def test_simulator_subtracts_and_exports(device, tmp_path):
    zyx, ids, _ = small_model()
    pdb = tmp_path / "model.pdb"
    write_pdb(pdb, zyx, ids)

    def simulator(**config):
        return Simulator(
            pdb_filepath=pdb,
            pixel_spacing=PIXEL,
            volume_shape=SHAPE,
            center_atoms=False,
            remove_hydrogens=False,
            simulator_config=SimulatorConfig(
                dose_start=0.0, dose_end=50.0, mtf_reference="k2_300kv", **config
            ),
        )

    sim = simulator()
    today, excluded = sim.run(device=device, return_excluded_volume=True)
    assert sim.excluded_volume is excluded
    scale = float(today.abs().max())
    assert torch.equal(simulator().run(device=device), today)

    water = simulator(subtract_solvent=True)
    template = water.run(device=device)
    torch.testing.assert_close(
        template, today - WATER_SOLVENT_DENSITY * excluded, rtol=0, atol=1e-6 * scale
    )

    water.export_to_mrc(
        tmp_path / "t.mrc", device=device, excluded_volume_filepath=tmp_path / "e.mrc"
    )
    t_file = mrcfile.read(tmp_path / "t.mrc")
    e_file = mrcfile.read(tmp_path / "e.mrc")
    tol = 1e-6 * scale
    np.testing.assert_allclose(t_file, template.cpu().numpy(), rtol=0, atol=tol)
    np.testing.assert_allclose(e_file, excluded.cpu().numpy(), rtol=0, atol=tol)


def test_cli_solvent_options(tmp_path):
    zyx, ids, _ = small_model()
    pdb = tmp_path / "model.pdb"
    write_pdb(pdb, zyx, ids)
    base = [
        "--pdb-filepath",
        str(pdb),
        "--pixel-spacing",
        str(PIXEL),
        "--volume-shape",
        *[str(s) for s in SHAPE],
        "--mtf-reference",
        "k2_300kv",
        "--apply-dqe",
    ]
    runner = CliRunner()

    help_text = runner.invoke(run_simulation_cli, ["--help"]).output
    for option in ("--subtract-solvent", "--solvent-density", "--excluded-volume-mrc"):
        assert option in help_text

    result = runner.invoke(
        run_simulation_cli,
        [*base, "--mrc-filepath", str(tmp_path / "x.mrc"), "--solvent-density", "0.7"],
    )
    assert result.exit_code == 2
    assert "--subtract-solvent" in result.output

    result = runner.invoke(
        run_simulation_cli,
        [
            *base,
            "--mrc-filepath",
            str(tmp_path / "today.mrc"),
            "--excluded-volume-mrc",
            str(tmp_path / "e.mrc"),
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        run_simulation_cli,
        [
            *base,
            "--mrc-filepath",
            str(tmp_path / "water.mrc"),
            "--subtract-solvent",
            "--solvent-density",
            "0.7",
        ],
    )
    assert result.exit_code == 0, result.output

    today = mrcfile.read(tmp_path / "today.mrc")
    excluded = mrcfile.read(tmp_path / "e.mrc")
    water = mrcfile.read(tmp_path / "water.mrc")
    np.testing.assert_allclose(
        water, today - 0.7 * excluded, rtol=0, atol=1e-6 * np.abs(today).max()
    )
