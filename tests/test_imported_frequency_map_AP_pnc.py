"""Read-only archive import verification. Athul Prem | _AP_pnc."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import numpy as np

from tools import import_lawrence_frequency_map_AP_pnc as importer
from tools import map_scattering_AP_pnc as mapper


INCIDENT_ROOTS = [19, 7]
REFLECTED_ROOTS = [20, 8]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_public_tables(directory):
    """Make a tiny signed, power-balanced CSV fixture for the archive reader."""
    reflected = (-0.3 + 0.4j, 0.5j)
    transmitted = (0.5 + math.sqrt(0.5) * 1j, -math.sqrt(0.5) + 0.5j)
    modes = [
        {"port_index": root, "k_real_rad_m": sign * (1e6 if index == 0 else -2e6), "k_imag_rad_m": 0.0,
         "direction": "+x" if sign == 1 else "-x", "propagating": True,
         "signed_power_W": sign, "relative_residual": 2e-15}
        for roots, sign in ((INCIDENT_ROOTS, 1), (REFLECTED_ROOTS, -1))
        for index, root in enumerate(roots)
    ]
    write_csv(directory / "port_modes_AP_pnc.csv", sorted(modes, key=lambda row: row["port_index"]))
    write_csv(directory / "scattering_AP_pnc.csv", [
        {"incident_mode_order": column, "frequency_GHz": 1.0, "transmission": 0.75, "reflection": 0.25,
         "power_balance": 0.0, "transmission_flux": 0.75, "reflection_flux": 0.25,
         "flux_power_balance": 0.0, "relative_residual": 2e-16}
        for column in (1, 2)
    ])
    channels = []
    for kind, roots, amplitudes in (("reflection", REFLECTED_ROOTS, reflected),
                                    ("transmission", INCIDENT_ROOTS, transmitted)):
        for row, outgoing in enumerate(roots):
            for column, incoming in enumerate(INCIDENT_ROOTS):
                amplitude = amplitudes[column] if row == column else 0j
                channels.append({"kind": kind, "incident_mode_order": column + 1, "incident_port_index": incoming,
                                 "outgoing_port_index": outgoing, "amplitude_real": amplitude.real,
                                 "amplitude_imag": amplitude.imag, "power_fraction": abs(amplitude) ** 2})
    write_csv(directory / "scattering_channels_AP_pnc.csv", channels)


def snapshot(directory):
    return {path.relative_to(directory): (path.stat().st_mtime_ns, path.read_bytes())
            for path in directory.rglob("*") if path.is_file()}


@contextmanager
def forbid_computation():
    read_bytes = Path.read_bytes

    def bounded_read(path):
        if path.suffix in (".npz", ".npy", ".vtu"):
            raise AssertionError(f"Read-only import buffered an entire scientific array: {path.name}")
        return read_bytes(path)

    with ExitStack() as stack:
        for owner, name in ((mapper.workflow, "build_mesh"), (mapper.workflow, "_assemble_system"),
                            (mapper.workflow, "assemble_port"), (mapper.workflow, "solve_port_modes"),
                            (mapper.workflow, "solve_scattering"), (mapper, "splu"), (np, "load")):
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError(f"Read-only import called {name}")))
        stack.enter_context(patch.object(Path, "read_bytes", bounded_read))
        yield


def acceptance_inputs():
    """Prepare small receipt records for the gate checks without a numerical replay."""
    equations = {"operator_csc_sha256": "a" * 64, "rhs_array_sha256": "b" * 64}
    solver = {
        **equations,
        "relative_target": 1e-10,
        "relative_residual_limit": 1e-9,
        "backward_residual_limit": 1e-9,
        "channels": [
            {"column": index, "incident_port_root": root, "info": 0, "target_met": True,
             "relative_residual": 3e-12, "relative_backward_residual": 2e-16}
            for index, root in enumerate(INCIDENT_ROOTS)
        ],
    }
    replay = {
        "execution_platform": "Windows",
        "accepted_drive_count": 2,
        "relative_l2_target": 1e-10,
        "relative_l2_residual_gate": 1e-9,
        "normwise_backward_residual_gate": 1e-9,
        "maximum_relative_l2_residual": 4e-12,
        "maximum_normwise_backward_residual": 2e-16,
        "channels": [
            {"incident_port_root": root, "relative_l2_residual": 4e-12, "normwise_backward_residual": 2e-16}
            for root in INCIDENT_ROOTS
        ],
    }
    evidence = {
        "direct_solve": {"independent_original_equation_replay": replay},
        "operator_stage": equations.copy(),
        "scattering_export": {"independent_nodal_force_flux_replay": {"execution_platform": "Windows",
                                                                    "maximum_flux_imbalance": 1e-14},
                              "staged_operator_and_rhs_reconstructed_exactly": True,
                              "public_fields_exactly_match_accepted_vectors": True,
                              "public_safe_wavenumber_displacement_and_force_arrays_match_saved_z6_section": True,
                              "phase_adjustment_applied": False},
    }
    return {"linear_solver": solver}, equations, evidence


def verified_archive_fixture(root):
    """Fixture at the publication boundary. Numerical files are opaque checksum inputs."""
    public = root / "public run"
    public.mkdir(parents=True)
    write_public_tables(public)
    for name in importer.PUBLIC_FILES:
        path = public / name
        if not path.exists():
            path.write_text(name, encoding="utf-8")

    def link(path):
        relative = path.relative_to(root).as_posix()
        return importer._verify_archived_link(root, relative, sha256(path.read_bytes()).hexdigest())

    outputs = {name: link(public / name) for name in importer.PUBLIC_FILES}
    summary = {"output_sha256": {name: record["sha256"] for name, record in outputs.items()},
               "frequency_hz": 1e9, "incident_side": "left", "incident_mode_indices": INCIDENT_ROOTS,
               "num_incident_channels": 2, "elapsed_seconds": 1.0, "scattering_resources": {}}
    summary_path = public / mapper.RUN_SUMMARY
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    thickness = json.loads(importer.THICKNESS_EVIDENCE.read_bytes())
    archive = {
        "configuration": {"analysis": {"kind": "scattering", "frequency_GHz": 1, "incident_side": "left"}},
        "shared": {"reference_planes_x_m": [0.0, 1.665e-5], "artifacts": {}},
        "receipt_links": {},
        "archived_execution": {"solve": {"slurm_job_id": "example",
                                        "implementation": {"git_commit": None, "platform": "Linux"}}},
        "verification_evidence": {},
        "mesh_pair_checks": importer._mesh_pair_evidence(thickness),
    }
    solve, stage, evidence = acceptance_inputs()
    channels, checks = importer._acceptance(solve, stage, evidence, INCIDENT_ROOTS)
    point = {"accepted": {"artifacts": {"public_summary": link(summary_path), "public_outputs": outputs}, **stage},
             "numerical_checks": checks, "incoming_channels": channels,
             "mode_identity": {"incident_roots": INCIDENT_ROOTS, "reflected_roots": REFLECTED_ROOTS,
                               "transmitted_roots": INCIDENT_ROOTS,
                               "cutoff_status": "No unresolved cutoff reported by the synthetic signed table.",
                               "mode_completeness_established": False}}
    return archive, point, public, summary


class ArchivedInputTests(unittest.TestCase):
    def test_rebase_preserves_linux_identity_and_checks_exact_local_bytes(self):
        with TemporaryDirectory() as temporary:
            root = (Path(temporary) / "relocated archive").resolve()
            relative = "saved-stage/K_AP_pnc.npz"
            file = root / relative
            file.parent.mkdir(parents=True)
            file.write_bytes(b"example stiffness")
            checksum = sha256(b"example stiffness").hexdigest()
            remote = (importer.LINUX_RUNS / relative).as_posix()
            saved = {"path": remote, "sha256": checksum, "bytes": len(b"example stiffness")}
            with forbid_computation():
                link = importer._verify_archived_link(root, relative, checksum, saved)
                self.assertEqual(link, {"path": relative, "original_linux_path": remote,
                                        "sha256": checksum, "bytes": saved["bytes"]})
                self.assertEqual(root / link["path"], file)

                wrong_phase = {**saved, "path": (importer.LINUX_RUNS / "other-stage/K_AP_pnc.npz").as_posix()}
                with self.assertRaisesRegex(ValueError, "Linux.*identity"):
                    importer._verify_archived_link(root, relative, checksum, wrong_phase)
                with self.assertRaisesRegex(ValueError, "relative archive artifact path"):
                    importer._verify_archived_link(root, "saved-stage/../saved-stage/K_AP_pnc.npz", checksum)
                file.write_bytes(b"changed stiffness")
                with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                    importer._verify_archived_link(root, relative, checksum, saved)
                file.unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    importer._verify_archived_link(root, relative, checksum, saved)

    def test_linked_mesh_safe_and_stage_must_have_the_same_physical_inputs(self):
        evidence = json.loads(importer.THICKNESS_EVIDENCE.read_bytes())
        mesh_hash = evidence["mesh_preflight"]["mesh_hash"]
        safe_evidence = evidence["safe_section"]
        left, right = evidence["selected_control"]["fixed_physical_end_planes_m"]
        configuration = {
            "geometry": {"length_nm": 16650, "width_nm": 1000, "thickness_nm": 200},
            "mesh": {"kind": "layered", "element_order": 2, "thickness_layers": 6},
            "material": {"C44_Pa": 66.8e9},
            "analysis": {"frequency_GHz": 1.0, "incident_side": "left"},
            "resources": {"max_dofs": 60000},
        }
        preflight = {"mesh_hash": mesh_hash, "configuration": deepcopy(configuration),
                     "geometry_verification": {
                         "port_end_planes": {"xleft": {"x_m": left}, "xright": {"x_m": right}}}}
        safe = {"configuration": deepcopy(configuration), "incident_port_roots": safe_evidence["positive_propagating_roots"],
                "backward_port_roots": safe_evidence["negative_propagating_roots"], "frequency_hz": 1e9,
                "trace_sha256": safe_evidence["trace_sha256"]}
        stage = {**deepcopy(safe), "mesh_hash": mesh_hash}
        stage["configuration"]["resources"]["max_dofs"] = 616239
        importer._verify_phase_inputs(preflight, safe, stage, evidence)

        altered_stage = deepcopy(stage)
        altered_stage["configuration"]["material"]["C44_Pa"] = 67e9
        with self.assertRaisesRegex(ValueError, "configuration"):
            importer._verify_phase_inputs(preflight, safe, altered_stage, evidence)
        altered_preflight = deepcopy(preflight)
        altered_preflight["geometry_verification"]["port_end_planes"]["xright"]["x_m"] = right - 1e-9
        with self.assertRaisesRegex(ValueError, "reference planes"):
            importer._verify_phase_inputs(altered_preflight, safe, stage, evidence)

    def test_publication_provenance_must_match_the_linked_linux_source(self):
        source = "src/phononic/scattering_AP_pnc.py"
        records = {
            phase: {"hostname": "test-node", "slurm_job_id": str(index),
                    "driver": {"path": f"/example/{phase}.py", "sha256": "b" * 64},
                    "implementation": {"platform": "Linux-test", "source_sha256": {source: "a" * 64}},
                    "numerical_thread_environment": {"OMP_NUM_THREADS": "1"}}
            for index, phase in enumerate(("safe", "stage", "solve", "export"), 1)
        }
        public = {
            "linux_assembly_and_solve": {**deepcopy(records["solve"]), "assembly": deepcopy(records["stage"])},
            "linux_safe": deepcopy(records["safe"]),
            "linux_export": deepcopy(records["export"]),
        }
        importer._verify_public_execution(public, records)
        public["linux_safe"]["implementation"]["source_sha256"][source] = "c" * 64
        with self.assertRaisesRegex(ValueError, "original safe receipt"):
            importer._verify_public_execution(public, records)

    def test_receipt_validation_rejects_different_equations_order_and_residuals(self):
        solve, stage, evidence = acceptance_inputs()
        channels, checks = importer._acceptance(solve, stage, evidence, INCIDENT_ROOTS)
        self.assertEqual([row["port_index"] for row in channels], INCIDENT_ROOTS)
        self.assertEqual(channels[0]["relative_l2_residual"], 4e-12)
        self.assertEqual(channels[0]["normwise_backward_residual"], 2e-16)
        self.assertEqual(checks["independent_nodal_force_flux_replay"]["maximum_flux_imbalance"], 1e-14)

        for case in ("different A", "different RHS", "reordered drives", "failed solver target",
                     "failed independent backward gate", "failed independent flux"):
            with self.subTest(case=case):
                changed_solve, changed_stage, changed_evidence = acceptance_inputs()
                solver = changed_solve["linear_solver"]
                if case == "different A":
                    changed_stage["operator_csc_sha256"] = "c" * 64
                elif case == "different RHS":
                    solver["rhs_array_sha256"] = "c" * 64
                elif case == "reordered drives":
                    for root, row in zip(reversed(INCIDENT_ROOTS), solver["channels"], strict=True):
                        row["incident_port_root"] = root
                elif case == "failed solver target":
                    solver["channels"][0]["relative_residual"] = 2e-10
                elif case == "failed independent backward gate":
                    replay = changed_evidence["direct_solve"]["independent_original_equation_replay"]
                    replay["channels"][0]["normwise_backward_residual"] = 2e-9
                else:
                    changed_evidence["scattering_export"]["independent_nodal_force_flux_replay"][
                        "maximum_flux_imbalance"] = 0.02
                with self.assertRaises(ValueError):
                    importer._acceptance(changed_solve, changed_stage, changed_evidence, INCIDENT_ROOTS)


class ImportedMapTests(unittest.TestCase):
    def test_publication_preserves_signed_modes_gates_and_skips_unsampled_frequencies(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "imported experiment"
            archive_root, output = root / "archive", root / "frequency map"
            fixture = verified_archive_fixture(archive_root)
            original_archive = snapshot(archive_root)

            def read_relocated(root):
                archive, point, public, summary = deepcopy(fixture)
                return archive, point, root / public.name, summary

            with (patch.object(importer, "_verify_archive", side_effect=read_relocated),
                  forbid_computation()):
                result = importer.import_lawrence_frequency_map(archive_root, output, frequencies_GHz=(0.9, 1.0, 1.1))
                retained_map = snapshot(output)
                reused = importer.import_lawrence_frequency_map(archive_root, output, frequencies_GHz=(0.9, 1.0, 1.1))
                with self.assertRaisesRegex(ValueError, "outside.*archive"):
                    importer.import_lawrence_frequency_map(archive_root, archive_root / "nested")
            self.assertEqual(snapshot(archive_root), original_archive)
            self.assertEqual(snapshot(output), retained_map)
            self.assertEqual(reused, result)
            self.assertEqual(result["execution_mode"], "archived_lawrence_read_only")
            self.assertFalse(result["computation_enabled"])
            self.assertEqual((output / result["archive_root"]).resolve(), archive_root)
            self.assertEqual([point["status"] for point in result["points"]], ["pending", "accepted", "pending"])
            self.assertTrue(all(point["spatial_convergence_established"] is False
                                and point["frequency_convergence_established"] is False for point in result["points"]))
            pairs = result["mesh_pair_checks"]
            self.assertEqual(set(pairs), {"in_plane_h100_h80_z4", "in_plane_h80_h60_z4",
                                          "joint_body_port_thickness_h80_z4_z6"})
            self.assertTrue(all(pair["mesh_pair_passed"] is False and pair["complex_passed"] is False
                                and pair["power_passed"] is True for pair in pairs.values()))
            self.assertEqual(result["shared"]["reference_planes_x_m"], [0.0, 1.665e-5])
            self.assertEqual(result["archived_execution"]["solve"]["implementation"],
                             {"git_commit": None, "platform": "Linux"})
            first, anchor, last = result["points"]
            self.assertNotIn("incoming_channels", first)
            self.assertNotIn("incoming_channels", last)
            self.assertEqual(anchor["track_from_previous"]["status"], "segment_start")
            self.assertEqual((output / unquote(anchor["report"])).resolve(), fixture[2] / mapper.REPORT)
            self.assertEqual(anchor["mode_identity"]["reflected_roots"], REFLECTED_ROOTS)
            self.assertEqual([row["port_index"] for row in anchor["incoming_channels"]], INCIDENT_ROOTS)
            self.assertEqual(anchor["incoming_channels"][0]["relative_l2_residual"], 4e-12)
            self.assertEqual(anchor["incoming_channels"][0]["normwise_backward_residual"], 2e-16)
            self.assertEqual(anchor["incoming_channels"][1]["k_rad_m"], -2e6)
            with (output / mapper.TABLE).open(encoding="utf-8", newline="") as table:
                rows = list(csv.DictReader(table))
            self.assertEqual(len(rows), 8)
            self.assertEqual({float(row["frequency_GHz"]) for row in rows}, {1.0})
            self.assertEqual({(row["kind"], int(row["outgoing_port_index"]), int(row["outgoing_channel"]))
                              for row in rows}, {("reflection", 20, 1), ("reflection", 8, 2),
                                                 ("transmission", 19, 1), ("transmission", 7, 2)})
            first_reflection = next(row for row in rows if row["kind"] == "reflection"
                                    and row["incident_port_index"] == "19" and row["outgoing_port_index"] == "20")
            self.assertEqual(complex(float(first_reflection["amplitude_real"]), float(first_reflection["amplitude_imag"])),
                             -0.3 + 0.4j)
            report = (output / mapper.REPORT).read_text(encoding="utf-8")
            self.assertIn("Pending", report)
            self.assertIn("16.65", report)
            self.assertEqual(report.count("· Fail"), 3)

            relocated = root.rename(root.with_name("relocated experiment"))
            archive_root, output = relocated / archive_root.name, relocated / output.name
            with (patch.object(importer, "_verify_archive", side_effect=read_relocated),
                  forbid_computation()):
                moved_map = importer.import_lawrence_frequency_map(archive_root, output, frequencies_GHz=(0.9, 1.0, 1.1))
            self.assertEqual(moved_map, result)
            self.assertEqual((output / moved_map["archive_root"]).resolve(), archive_root)
            self.assertEqual(snapshot(archive_root), original_archive)
            self.assertEqual(snapshot(output), retained_map)

            (output / mapper.TABLE).write_text("changed after publication", encoding="utf-8")
            changed_map = snapshot(output)
            with (patch.object(importer, "_verify_archive", side_effect=read_relocated),
                  forbid_computation(), self.assertRaisesRegex(ValueError, "SHA-256 changed")):
                importer.import_lawrence_frequency_map(archive_root, output, frequencies_GHz=(0.9, 1.0, 1.1))
            self.assertEqual(snapshot(output), changed_map)


if __name__ == "__main__":
    unittest.main()
