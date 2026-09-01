#!/usr/bin/env python3
"""Benchmark ttsim3d vs teamtomo potential backends."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ttsim3d.models import Simulator, SimulatorConfig


@dataclass
class BenchmarkResult:
  """One backend run and optional comparison against a reference volume."""

  label: str
  wall_time_s: float
  peak_memory_gb: float | None
  volume: torch.Tensor
  spacing: float
  pearson_vs_ref: float | None = None
  scale_vs_ref: float | None = None
  rel_rmse_vs_ref: float | None = None


def _sync(device: str) -> None:
  if device.startswith("cuda"):
    torch.cuda.synchronize()


def _peak_memory_gb(device: str) -> float | None:
  if not device.startswith("cuda"):
    return None
  return torch.cuda.max_memory_allocated() / 1e9


def _compare_volumes(
  reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[float, float, float]:
  ref = reference.detach().float().flatten()
  cand = candidate.detach().float().flatten()
  ref_centered = ref - ref.mean()
  cand_centered = cand - cand.mean()
  pearson = float(
    (ref_centered @ cand_centered)
    / (ref_centered.norm() * cand_centered.norm() + 1e-12)
  )
  scale = float((ref @ cand) / (cand @ cand + 1e-12))
  residual = ref - scale * cand
  rel_rmse = float(residual.norm() / (ref.norm() + 1e-12))
  return pearson, scale, rel_rmse


def run_case(
  *,
  pdb_filepath: Path,
  pixel_spacing: float,
  volume_shape: tuple[int, int, int],
  device: str,
  backend: str,
  bonded: bool,
  upsampling: int,
) -> BenchmarkResult:
  label = f"{backend}_{'bonded' if bonded else 'elemental'}"
  config = SimulatorConfig(
    apply_dose_weighting=False,
    apply_dqe=False,
    store_volume=False,
    upsampling=upsampling,
    use_bonded_scattering_factors=bonded,
    potential_backend=backend,  # type: ignore[arg-type]
  )
  sim = Simulator(
    pdb_filepath=pdb_filepath,
    pixel_spacing=pixel_spacing,
    volume_shape=volume_shape,
    simulator_config=config,
    remove_hydrogens=True,
  )

  if device.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats()

  _sync(device)
  start = time.perf_counter()
  volume, spacing = sim.run(device=device)
  _sync(device)
  elapsed = time.perf_counter() - start

  return BenchmarkResult(
    label=label,
    wall_time_s=elapsed,
    peak_memory_gb=_peak_memory_gb(device),
    volume=volume.detach().cpu(),
    spacing=float(spacing),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "pdb_filepath",
    type=Path,
    help="Input structure file (PDB/mmCIF).",
  )
  parser.add_argument("--pixel-spacing", type=float, default=2.0)
  parser.add_argument("--shape", type=int, default=64, help="Cubic volume size.")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--upsampling",
    type=int,
    default=1,
    help="Upsampling for the ttsim3d backend only.",
  )
  parser.add_argument(
    "--reference",
    choices=("ttsim3d", "teamtomo"),
    default="ttsim3d",
    help="Backend used as the reference for correlation metrics.",
  )
  args = parser.parse_args()

  volume_shape = (args.shape, args.shape, args.shape)
  cases = [
    ("ttsim3d", False),
    ("ttsim3d", True),
    ("teamtomo", False),
    ("teamtomo", True),
  ]

  results: list[BenchmarkResult] = []
  for backend, bonded in cases:
    print(f"Running {backend} ({'bonded' if bonded else 'elemental'})...")
    result = run_case(
      pdb_filepath=args.pdb_filepath,
      pixel_spacing=args.pixel_spacing,
      volume_shape=volume_shape,
      device=args.device,
      backend=backend,
      bonded=bonded,
      upsampling=args.upsampling,
    )
    results.append(result)

  for bonded in (False, True):
    ref = next(
      r
      for r in results
      if r.label == f"{args.reference}_{'bonded' if bonded else 'elemental'}"
    )
    for result in results:
      if result.label.endswith("bonded" if bonded else "elemental"):
        pearson, scale, rel_rmse = _compare_volumes(ref.volume, result.volume)
        result.pearson_vs_ref = pearson
        result.scale_vs_ref = scale
        result.rel_rmse_vs_ref = rel_rmse

  print()
  print(
    f"{'label':<22} {'time_s':>8} {'mem_GB':>8} {'spacing':>8} "
    f"{'pearson':>8} {'scale':>8} {'rel_rmse':>8}"
  )
  for result in results:
    mem = "" if result.peak_memory_gb is None else f"{result.peak_memory_gb:8.2f}"
    pearson = (
      ""
      if result.pearson_vs_ref is None
      else f"{result.pearson_vs_ref:8.4f}"
    )
    scale = "" if result.scale_vs_ref is None else f"{result.scale_vs_ref:8.4f}"
    rel_rmse = (
      ""
      if result.rel_rmse_vs_ref is None
      else f"{result.rel_rmse_vs_ref:8.4f}"
    )
    print(
      f"{result.label:<22} {result.wall_time_s:8.2f} {mem:>8} "
      f"{result.spacing:8.4f} {pearson:>8} {scale:>8} {rel_rmse:>8}"
    )


if __name__ == "__main__":
  main()
