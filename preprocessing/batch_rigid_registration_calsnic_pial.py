#!/usr/bin/env python3
"""
Batch rigid registration for CALSNIC pial meshes (already eTIV-normalized).

This script registers three normalized mesh sets:
1) Whole brain
2) Left hemisphere
3) Right hemisphere

Inputs are expected under:
  /home/jakaria/CALSNIC/calsnic_pial_surface/mesh_dataset/
    pial_surface_volume_normalized_for_sex
    pial_surface_L_volume_normalized_for_sex
    pial_surface_R_volume_normalized_for_sex

Outputs are written to sibling folders with suffix "_rigid_reg":
    *_volume_normalized_for_sex_rigid_reg

Key guarantees:
- Rigid transform only (rotation + translation), so geometric volume should be preserved.
- Vertex ordering and face connectivity are preserved in output.
- If measured output volume deviates by > 1% (default), the script:
  1) prints a warning,
  2) saves it in an exceedance CSV,
  3) applies isotropic correction scaling to match input volume,
  4) records scaling factor in report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import trimesh


@dataclass(frozen=True)
class MeshSet:
    name: str
    input_dirname: str
    output_dirname: str


MESH_SETS: Tuple[MeshSet, ...] = (
    MeshSet(
        name="whole",
        input_dirname="pial_surface_volume_normalized_for_sex",
        output_dirname="pial_surface_volume_normalized_for_sex_rigid_reg",
    ),
    MeshSet(
        name="left",
        input_dirname="pial_surface_L_volume_normalized_for_sex",
        output_dirname="pial_surface_L_volume_normalized_for_sex_rigid_reg",
    ),
    MeshSet(
        name="right",
        input_dirname="pial_surface_R_volume_normalized_for_sex",
        output_dirname="pial_surface_R_volume_normalized_for_sex_rigid_reg",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rigidly register normalized CALSNIC pial meshes for whole/left/right, "
            "while checking and enforcing volume consistency."
        )
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=Path("/home/jakaria/CALSNIC/calsnic_pial_surface/mesh_dataset"),
        help="Root directory containing normalized input mesh folders.",
    )
    parser.add_argument(
        "--volume-tolerance",
        type=float,
        default=0.01,
        help="Relative volume tolerance (default 1%%).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Small epsilon for numerical stability.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("/home/jakaria/CALSNIC/rigid_registration_volume_report.csv"),
        help="Path to save full registration/volume report CSV.",
    )
    parser.add_argument(
        "--exceed-csv",
        type=Path,
        default=Path("/home/jakaria/CALSNIC/rigid_registration_volume_exceeds_1pct.csv"),
        help="Path to save rows where rigid volume deviation exceeded tolerance.",
    )
    parser.add_argument(
        "--max-files-per-set",
        type=int,
        default=None,
        help="Optional debug limit: process only first N files per set.",
    )
    return parser.parse_args()


def mesh_volume_abs(mesh: trimesh.Trimesh) -> float:
    return abs(float(mesh.volume))


def relative_error(a: float, b: float, epsilon: float) -> float:
    return abs(a - b) / max(abs(b), epsilon)


def subject_id_from_path(ply_path: Path) -> str:
    sid = ply_path.stem
    if sid.endswith("_L") or sid.endswith("_R"):
        sid = sid[:-2]
    return sid


def load_mesh(ply_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(ply_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Loaded object is not a mesh: {ply_path}")
    return mesh


def kabsch_rigid_transform(
    src_vertices: np.ndarray, dst_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve rigid transform (R, t) such that:
      dst ~ src @ R.T + t
    using Kabsch with det(R)=+1 (no reflection).
    """
    src = np.asarray(src_vertices, dtype=np.float64)
    dst = np.asarray(dst_vertices, dtype=np.float64)

    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    src_centered = src - c_src
    dst_centered = dst - c_dst

    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    t = c_dst - (c_src @ r.T)
    return r, t


def apply_rigid(vertices: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return vertices @ r.T + t


def isotropic_scale_about_centroid(vertices: np.ndarray, scale: float) -> np.ndarray:
    c = vertices.mean(axis=0)
    return (vertices - c) * scale + c


def choose_reference_medoid_approx(ply_files: List[Path]) -> Tuple[int, str]:
    """
    Approximate medoid selection:
    - Compute mean centered shape
    - Choose mesh with minimum RMS distance to that mean centered shape

    This avoids O(N^2) pairwise distance while still choosing a representative mesh.
    """
    sum_centered = None
    n = 0

    for ply in ply_files:
        v = np.asarray(load_mesh(ply).vertices, dtype=np.float64)
        vc = v - v.mean(axis=0)
        if sum_centered is None:
            sum_centered = np.zeros_like(vc, dtype=np.float64)
        sum_centered += vc
        n += 1

    if sum_centered is None or n == 0:
        raise RuntimeError("No meshes available for reference selection.")

    mean_centered = sum_centered / float(n)

    best_idx = 0
    best_score = np.inf
    for idx, ply in enumerate(ply_files):
        v = np.asarray(load_mesh(ply).vertices, dtype=np.float64)
        vc = v - v.mean(axis=0)
        score = float(np.mean((vc - mean_centered) ** 2))
        if score < best_score:
            best_score = score
            best_idx = idx

    ref_subject = subject_id_from_path(ply_files[best_idx])
    return best_idx, ref_subject


def process_one_set(
    mesh_set: MeshSet,
    mesh_root: Path,
    volume_tolerance: float,
    epsilon: float,
    max_files_per_set: Optional[int],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    input_dir = mesh_root / mesh_set.input_dirname
    output_dir = mesh_root / mesh_set.output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(input_dir.glob("*.ply"))
    if max_files_per_set is not None:
        ply_files = ply_files[: max_files_per_set]

    if len(ply_files) == 0:
        raise RuntimeError(f"No PLY files found in {input_dir}")

    print("\n" + "=" * 90)
    print(f"SET: {mesh_set.name.upper()}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(ply_files)}")

    ref_idx, ref_subject = choose_reference_medoid_approx(ply_files)
    ref_path = ply_files[ref_idx]
    ref_mesh = load_mesh(ref_path)
    ref_vertices = np.asarray(ref_mesh.vertices, dtype=np.float64)
    ref_faces = np.asarray(ref_mesh.faces, dtype=np.int64)
    ref_v_count = int(ref_vertices.shape[0])
    ref_f_count = int(ref_faces.shape[0])

    print(f"Reference (approx medoid): {ref_subject} [{ref_path.name}]")
    print(f"Reference topology: V={ref_v_count}, F={ref_f_count}")

    rows: List[Dict[str, object]] = []
    n_ok = 0
    n_corrected = 0
    n_failed = 0
    n_exceed_before = 0
    n_topology_mismatch = 0

    for i, src_path in enumerate(ply_files, start=1):
        sid = subject_id_from_path(src_path)
        out_path = output_dir / src_path.name
        row: Dict[str, object] = {
            "set_name": mesh_set.name,
            "subject_id": sid,
            "input_path": str(src_path),
            "output_path": str(out_path),
            "reference_subject": ref_subject,
            "reference_file": ref_path.name,
        }

        try:
            mesh_in = load_mesh(src_path)
            vertices_in = np.asarray(mesh_in.vertices, dtype=np.float64)
            faces_in = np.asarray(mesh_in.faces, dtype=np.int64)

            topo_match_ref = (
                vertices_in.shape[0] == ref_v_count
                and faces_in.shape == ref_faces.shape
                and np.array_equal(faces_in, ref_faces)
            )
            if not topo_match_ref:
                n_topology_mismatch += 1

            row["n_vertices"] = int(vertices_in.shape[0])
            row["n_faces"] = int(faces_in.shape[0])
            row["topology_matches_reference_input"] = bool(topo_match_ref)

            # Rigid alignment (rotation + translation only)
            r, t = kabsch_rigid_transform(vertices_in, ref_vertices)
            vertices_rigid = apply_rigid(vertices_in, r, t)

            mesh_rigid = trimesh.Trimesh(vertices=vertices_rigid, faces=faces_in, process=False)
            vol_in = mesh_volume_abs(mesh_in)
            vol_rigid = mesh_volume_abs(mesh_rigid)
            err_rigid = relative_error(vol_rigid, vol_in, epsilon)

            row["volume_input"] = vol_in
            row["volume_after_rigid"] = vol_rigid
            row["rel_error_after_rigid"] = err_rigid
            row["volume_scale_applied"] = False
            row["scale_factor"] = 1.0

            if err_rigid > volume_tolerance:
                n_exceed_before += 1
                print(
                    f"[WARN >{100.0*volume_tolerance:.1f}%] "
                    f"{mesh_set.name}/{sid}: rigid vol err={err_rigid:.4%}"
                )
                if vol_rigid <= epsilon:
                    raise RuntimeError(
                        f"Rigid output volume too small for correction: {vol_rigid}"
                    )
                scale = (vol_in / vol_rigid) ** (1.0 / 3.0)
                vertices_scaled = isotropic_scale_about_centroid(vertices_rigid, scale)
                mesh_final = trimesh.Trimesh(
                    vertices=vertices_scaled, faces=faces_in, process=False
                )
                row["volume_scale_applied"] = True
                row["scale_factor"] = float(scale)
                n_corrected += 1
            else:
                mesh_final = mesh_rigid

            mesh_final.export(out_path)

            # Verify output
            mesh_out = load_mesh(out_path)
            vertices_out = np.asarray(mesh_out.vertices, dtype=np.float64)
            faces_out = np.asarray(mesh_out.faces, dtype=np.int64)

            topo_preserved = (
                vertices_out.shape == vertices_in.shape
                and faces_out.shape == faces_in.shape
                and np.array_equal(faces_out, faces_in)
            )
            row["topology_preserved_output"] = bool(topo_preserved)

            vol_out = mesh_volume_abs(mesh_out)
            err_final = relative_error(vol_out, vol_in, epsilon)
            row["volume_output_final"] = vol_out
            row["rel_error_final"] = err_final
            row["status"] = "ok"
            row["message"] = ""

            if not topo_preserved:
                row["status"] = "error_topology_not_preserved"
                row["message"] = (
                    "Output topology mismatch: vertex/face count or face indices changed."
                )
                n_failed += 1
            else:
                n_ok += 1

        except Exception as exc:
            row["status"] = "error"
            row["message"] = str(exc)
            n_failed += 1
            print(f"[ERROR] {mesh_set.name}/{sid}: {exc}")

        rows.append(row)
        if i % 25 == 0 or i == len(ply_files):
            print(f"Processed {i}/{len(ply_files)}")

    # Save reference info for this set
    ref_file = output_dir / "reference_medoid.txt"
    with ref_file.open("w", encoding="utf-8") as f:
        f.write(f"CALSNIC PIAL {mesh_set.name.upper()} RIGID REGISTRATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Reference selection: approximate medoid (closest to mean-centered shape)\n")
        f.write(f"Reference subject: {ref_subject}\n")
        f.write(f"Reference file: {ref_path.name}\n")
        f.write(f"Input files: {len(ply_files)}\n")
        f.write(f"Input topology: vertices={ref_v_count}, faces={ref_f_count}\n")
        f.write(f"Volume tolerance: {volume_tolerance:.6f} ({100.0*volume_tolerance:.2f}%)\n")
        f.write("\n")
        f.write("Notes:\n")
        f.write("- Rigid registration preserves distances and should preserve volume.\n")
        f.write(
            "- Any case with rigid volume error above tolerance is corrected by isotropic scaling.\n"
        )
        f.write("- Vertex indexing and face connectivity are preserved in output files.\n")

    summary = {
        "set_name": mesh_set.name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "reference_subject": ref_subject,
        "reference_file": ref_path.name,
        "total_files": len(ply_files),
        "ok": n_ok,
        "corrected_over_tolerance": n_corrected,
        "failed": n_failed,
        "exceeded_before_correction": n_exceed_before,
        "topology_mismatch_input_vs_ref": n_topology_mismatch,
        "reference_info_file": str(ref_file),
    }
    return rows, summary


def main() -> None:
    args = parse_args()

    print("=" * 90)
    print("CALSNIC PIAL BATCH RIGID REGISTRATION (NORMALIZED MESHES)")
    print("=" * 90)
    print(f"Mesh root:           {args.mesh_root}")
    print(f"Volume tolerance:    {args.volume_tolerance:.6f} ({100.0*args.volume_tolerance:.2f}%)")
    print(f"Full report CSV:     {args.report_csv}")
    print(f"Exceedance report:   {args.exceed_csv}")
    if args.max_files_per_set is not None:
        print(f"Debug file limit:    {args.max_files_per_set}")

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []

    for mesh_set in MESH_SETS:
        rows, summary = process_one_set(
            mesh_set=mesh_set,
            mesh_root=args.mesh_root,
            volume_tolerance=args.volume_tolerance,
            epsilon=args.epsilon,
            max_files_per_set=args.max_files_per_set,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    report_df = pd.DataFrame(all_rows)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.report_csv, index=False)

    exceed_df = report_df[
        report_df["rel_error_after_rigid"].astype(float) > float(args.volume_tolerance)
    ].copy()
    args.exceed_csv.parent.mkdir(parents=True, exist_ok=True)
    exceed_df.to_csv(args.exceed_csv, index=False)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for s in summaries:
        print(f"\nSET: {s['set_name'].upper()}")
        print(f"  Input dir:   {s['input_dir']}")
        print(f"  Output dir:  {s['output_dir']}")
        print(f"  Reference:   {s['reference_subject']} ({s['reference_file']})")
        print(f"  Total files: {s['total_files']}")
        print(f"  OK:          {s['ok']}")
        print(f"  Corrected:   {s['corrected_over_tolerance']}")
        print(f"  Failed:      {s['failed']}")
        print(f"  >tol before correction: {s['exceeded_before_correction']}")
        print(f"  Input topology mismatch vs ref: {s['topology_mismatch_input_vs_ref']}")
        print(f"  Ref info:    {s['reference_info_file']}")

    print("\nReport saved:", args.report_csv)
    print("Exceedance saved:", args.exceed_csv)
    print("Total records:", len(report_df))
    print("Exceeded tolerance (before correction):", len(exceed_df))

    # Additional global integrity checks for correspondence and final volume
    if "topology_preserved_output" in report_df.columns:
        topo_ok = int(report_df["topology_preserved_output"].fillna(False).sum())
        print(f"Topology preserved rows: {topo_ok}/{len(report_df)}")
    if "rel_error_final" in report_df.columns:
        rel = pd.to_numeric(report_df["rel_error_final"], errors="coerce").dropna()
        if len(rel) > 0:
            print(
                f"Final relative volume error: max={rel.max():.3e}, mean={rel.mean():.3e}"
            )


if __name__ == "__main__":
    main()
