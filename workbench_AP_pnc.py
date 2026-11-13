#!/usr/bin/env python3
"""Desktop workbench for configured 3D beam studies. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import codecs
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
from tempfile import mkdtemp

import psutil

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
    from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QTextCursor
    from PySide6.QtWidgets import (
        QApplication, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
        QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
    )
except ImportError:
    raise SystemExit("Install requirements_ui_AP_pnc.txt to use the desktop workbench.") from None


ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "tools/workbench_worker_AP_pnc.py"
PRESETS = (
    ("Straight beam · finite modes", "straight_beam_AP_pnc.json"),
    ("Straight guide · scattering", "straight_scattering_AP_pnc.json"),
    ("Two-hole beam · scattering", "perforated_beam_AP_pnc.json"),
    ("Two-hole phase reference · scattering", "perforated_phase_reference_AP_pnc.json"),
    ("Historical 26-hole device · pilot", "historical_26hole_AP_pnc.json"),
)
ANALYSES = {"modes": "Finite-body modes", "bloch": "Bloch dispersion", "scattering": "Matched-port scattering"}
BOUNDARIES = {"clamped_free": "Clamped–free", "clamped_clamped": "Clamped at both ends", "free": "Free",
              "periodic_x": "Periodic along x", "matched_ports": "Matched ports"}
FIELDS = (
    ("name", "Study name"), ("length_nm", "Length (nm)"), ("width_nm", "Width (nm)"),
    ("thickness_nm", "Thickness (nm)"), ("frequency_GHz", "Frequency (GHz)"),
    ("num_modes", "Requested modes"), ("k_points", "Wavevector samples"),
    ("cells", "Mesh cells (x, y, z)"), ("mesh_size_nm", "Body mesh size (nm)"),
    ("port_size_nm", "Port mesh size (nm)"), ("circle_resolution", "Circle segments"),
    ("max_dofs", "Body DOF cap"), ("max_port_dofs", "Port DOF cap"),
)
NUMERICAL_FIELDS = {
    "geometry": ("length_nm", "width_nm", "thickness_nm"),
    "analysis": ("frequency_GHz", "num_modes", "k_points"),
    "mesh": ("mesh_size_nm", "port_size_nm", "circle_resolution"),
    "resources": ("max_dofs", "max_port_dofs"),
}
WHOLE_FIELDS = {"num_modes", "k_points", "circle_resolution", "max_dofs", "max_port_dofs"}


def _read_configuration(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Select a version 1 simulation configuration.")
    for section in ("geometry", "mesh", "material", "analysis", "resources"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"The simulation configuration requires a {section} object.")
    kind = config["analysis"].get("kind")
    if not isinstance(kind, str) or kind not in ANALYSES:
        raise ValueError("Select a finite-mode, Bloch, or scattering configuration.")
    boundary = config["analysis"].get("boundary")
    if not isinstance(boundary, str) or boundary not in BOUNDARIES:
        raise ValueError("Select a supported boundary condition.")
    if config["mesh"].get("kind", "structured") not in ("structured", "gmsh"):
        raise ValueError("The mesh kind must be structured or gmsh.")
    pattern = config["geometry"].get("pattern")
    if pattern is not None and not isinstance(pattern, dict):
        raise ValueError("The hole pattern must be a configuration object.")
    return config


class Workbench(QMainWindow):
    def __init__(self, output_root: Path | None = None):
        super().__init__()
        if (sys.platform == "win32" and QApplication.platformName() == "offscreen"
                and not QFontDatabase.families()):
            # Windows' offscreen plugin may not discover any system fonts.
            font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf"
            if font_path.is_file():
                families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font_path)))
                if families:
                    QApplication.setFont(QFont(families[0], 10))
        self.output_root = Path(output_root or ROOT / "results/runs").resolve()
        self.config = {}
        self.result_directory = None
        self.report_path = None
        self.action = None
        self.response_path = None
        self.stopping = False
        self.stopping_processes = []
        self.stop_timer = QTimer(self)
        self.stop_timer.setInterval(100)
        self.stop_timer.timeout.connect(self._check_stopped)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_log)
        self.process.started.connect(lambda: self.stop_button.setEnabled(not self.stopping))
        self.process.finished.connect(self._job_finished)
        self.process.errorOccurred.connect(self._process_error)

        self.setWindowTitle("Phononic Nanobeam Design Lab")
        self.resize(1000, 750)
        body = QWidget()
        self.setCentralWidget(body)
        main = QVBoxLayout(body)
        main.setContentsMargins(22, 16, 22, 18)
        heading = QLabel("Phononic Nanobeam Design Lab")
        heading.setStyleSheet("font-size: 23px; font-weight: 600;")
        main.addWidget(heading)

        choices = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.setAccessibleName("Simulation preset")
        for label, _ in PRESETS:
            self.preset.addItem(label)
        self.preset.addItem("Loaded configuration")
        self.load_button = QPushButton("Load JSON…")
        self.save_button = QPushButton("Save JSON…")
        choices.addWidget(QLabel("Preset"))
        choices.addWidget(self.preset, 1)
        choices.addWidget(self.load_button)
        choices.addWidget(self.save_button)
        main.addLayout(choices)

        columns = QHBoxLayout()
        main.addLayout(columns)
        self.form = QGroupBox("Study inputs")
        self.form_layout = QFormLayout(self.form)
        self.fields = {}
        for key, label in FIELDS:
            field = QLineEdit()
            field.setAccessibleName(label)
            field.textEdited.connect(self._inputs_changed)
            self.fields[key] = field
            self.form_layout.addRow(label, field)
        columns.addWidget(self.form, 1)

        metadata = QGroupBox("Model and provenance")
        metadata_layout = QVBoxLayout(metadata)
        self.provenance = QLabel()
        self.provenance.setTextFormat(Qt.TextFormat.PlainText)
        self.provenance.setWordWrap(True)
        self.provenance.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        metadata_layout.addWidget(self.provenance)
        metadata_layout.addStretch()
        columns.addWidget(metadata, 1)

        actions = QHBoxLayout()
        self.mesh_button = QPushButton("Check mesh")
        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet("font-weight: 600; padding-left: 15px; padding-right: 15px;")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.report_button = QPushButton("Open report")
        self.folder_button = QPushButton("Open folder")
        for button in (self.mesh_button, self.run_button, self.stop_button, self.report_button, self.folder_button):
            actions.addWidget(button)
        actions.addStretch()
        main.addLayout(actions)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        main.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setAccessibleName("Worker messages")
        self.log.setPlaceholderText("Worker messages")
        self.log.setMaximumBlockCount(1500)
        main.addWidget(self.log, 1)

        self.preset.currentIndexChanged.connect(self._select_preset)
        self.load_button.clicked.connect(self._choose_load)
        self.save_button.clicked.connect(self._choose_save)
        self.mesh_button.clicked.connect(lambda: self.start_job("mesh"))
        self.run_button.clicked.connect(lambda: self.start_job("run"))
        self.stop_button.clicked.connect(self.stop_job)
        self.report_button.clicked.connect(lambda: self._open(self.report_path))
        self.folder_button.clicked.connect(lambda: self._open(self.result_directory))
        self._select_preset(0)

    def _select_preset(self, index: int) -> None:
        if 0 <= index < len(PRESETS):
            self.load_configuration(ROOT / "configs" / PRESETS[index][1], index)

    def _choose_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load simulation configuration", str(ROOT / "configs"), "JSON (*.json)")
        if path:
            self.load_configuration(Path(path))

    def load_configuration(self, path: Path, index: int = len(PRESETS)) -> bool:
        if self.action is not None:
            return False
        try:
            config = _read_configuration(path)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not load configuration. {exc}")
            return False
        self.config = config
        was_blocked = self.preset.blockSignals(True)
        self.preset.setCurrentIndex(index)
        self.preset.blockSignals(was_blocked)
        mesh = config["mesh"]
        analysis = config["analysis"]
        gmsh = mesh.get("kind", "structured") == "gmsh"
        scattering = analysis["kind"] == "scattering"
        self.visible_fields = {"name", "length_nm", "width_nm", "thickness_nm", "max_dofs"}
        self.visible_fields.update(("mesh_size_nm", "port_size_nm", "circle_resolution") if gmsh else ("cells",))
        self.visible_fields.update(("frequency_GHz",) if scattering else ("num_modes",))
        if analysis["kind"] == "bloch":
            self.visible_fields.add("k_points")
        if gmsh or scattering:
            self.visible_fields.add("max_port_dofs")
        values = {"name": config.get("name", "straight_beam"), "cells": "", "k_points": 9, "circle_resolution": 24,
                  "max_port_dofs": 200}
        for section, keys in NUMERICAL_FIELDS.items():
            values.update({key: config[section][key] for key in keys if key in config[section]})
        if isinstance(mesh.get("cells"), (list, tuple)):
            values["cells"] = ", ".join(map(str, mesh["cells"]))
        for key, field in self.fields.items():
            field.setText(str(values.get(key, "")))
            self.form_layout.setRowVisible(field, key in self.visible_fields)
        self._show_provenance(path)
        self.log.clear()
        self._clear_result()
        self.status.setText("Configuration ready. Check the mesh or run the study.")
        return True

    def _show_provenance(self, path: Path) -> None:
        material, analysis, mesh = (self.config[key] for key in ("material", "analysis", "mesh"))
        boundary = BOUNDARIES.get(analysis.get("boundary"), analysis.get("boundary", "unspecified"))
        lines = [f"Analysis: {ANALYSES[analysis['kind']]} · {boundary}"]
        if analysis["kind"] == "scattering":
            lines.append(f"Incident side: {analysis.get('incident_side', 'left')}")
        lines.append(f"Displacement element order: {mesh.get('element_order', 'unspecified')}")
        lines.append(f"Configuration: {path}")
        pattern = self.config["geometry"].get("pattern")
        if pattern is None:
            lines.append("Hole pattern: uniform beam")
        else:
            lines.append(f"Hole table: {pattern.get('csv', 'unspecified')}")
            lines.append(f"Hole table x offset: {pattern.get('x_offset_nm', 'unspecified')} nm")
            lines.append("Hole positions and diameters stay fixed when beam dimensions change.")
        lines.extend(("", f"Material: {material.get('name', 'unspecified')}",
                      f"C11 / C12 / C44 (Pa): {material.get('C11_Pa')} / {material.get('C12_Pa')} / {material.get('C44_Pa')}",
                      f"Density (kg/m³): {material.get('density_kg_m3')}",
                      f"Crystal-to-lab rotation: {material.get('crystal_rotation')}",
                      f"Temperature label: {material.get('temperature_K')} K · {material.get('temperature_policy')}"))
        if material.get("temperature_policy") == "fixed_property_control":
            lines.append("The temperature label does not change these material properties.")
        self.provenance.setText("\n".join(lines))

    def edited_config(self) -> dict:
        config = deepcopy(self.config)
        name = self.fields["name"].text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("Study name must use 1–80 letters, digits, underscores, or hyphens.")
        config["name"] = name
        for section, keys in NUMERICAL_FIELDS.items():
            for key in keys:
                if key not in self.visible_fields:
                    continue
                field = self.fields[key]
                text = field.text().strip()
                try:
                    value = int(text) if key in WHOLE_FIELDS else float(text)
                    valid = value > 0 and math.isfinite(value)
                except (ValueError, OverflowError):
                    value = 0
                    valid = False
                if not valid:
                    unit = "whole number" if key in WHOLE_FIELDS else "number"
                    raise ValueError(f"{field.accessibleName()} must be a positive {unit}.")
                original = config[section].get(key)
                config[section][key] = original if type(original) in (int, float) and original == value else value
        if "cells" in self.visible_fields:
            try:
                cells = [int(value) for value in re.split(r"[,\s]+", self.fields["cells"].text().strip())]
            except ValueError:
                cells = []
            if len(cells) != 3 or min(cells) <= 0:
                raise ValueError("Mesh cells must contain three positive whole numbers.")
            config["mesh"]["cells"] = cells
        return config

    def _choose_save(self) -> None:
        try:
            config = self.edited_config()
            suggested = str(self.output_root / f"{config['name']}_AP_pnc.json")
            path, _ = QFileDialog.getSaveFileName(self, "Save simulation configuration", suggested, "JSON (*.json)")
            if path:
                self._write_config(Path(path), config)
                self.status.setText(f"Configuration saved to {path}")
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not save configuration. {exc}")

    @staticmethod
    def _write_config(path: Path, config: dict) -> None:
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    def _clear_result(self) -> None:
        self.result_directory = None
        self.report_path = None
        self.report_button.setEnabled(False)
        self.folder_button.setEnabled(False)

    def _inputs_changed(self) -> None:
        self._clear_result()
        self.status.setText("Inputs changed. Check the mesh or run the study.")

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.preset, self.load_button, self.save_button, self.form, self.mesh_button, self.run_button):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy and not self.stopping and self.process.state() == QProcess.ProcessState.Running)

    def start_job(self, action: str) -> bool:
        if self.action is not None:
            return False
        self._clear_result()
        try:
            config = self.edited_config()
            jobs = self.output_root / "_workbench"
            jobs.mkdir(parents=True, exist_ok=True)
            directory = Path(mkdtemp(prefix="job-", dir=jobs))
            config_path = directory / "configuration_AP_pnc.json"
            self._write_config(config_path, config)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not start the study. {exc}")
            return False
        self.response_path = directory / "response_AP_pnc.json"
        self.log_path = directory / "worker_AP_pnc.log"
        self.action = action
        self.log.clear()
        self.log_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._set_busy(True)
        self.status.setText("Checking the mesh…" if action == "mesh" else "Running the study…")
        self.process.start(sys.executable, ["-u", "-B", str(WORKER), "--config", str(config_path),
                                            "--action", action, "--output-root", str(self.output_root),
                                            "--response", str(self.response_path)])
        return True

    def _read_log(self, final: bool = False) -> None:
        data = bytes(self.process.readAllStandardOutput())
        if data:
            try:
                with self.log_path.open("ab") as stream:
                    stream.write(data)
            except OSError as exc:
                self.status.setText(f"Could not save the worker log. {exc}")
        text = self.log_decoder.decode(data, final=final)
        if text:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self.action is not None:
            self.status.setText(f"Could not start the study worker. {self.process.errorString()}")
            self._finish_job()

    def _finish_job(self) -> None:
        self.action = None
        self.stopping = False
        self.stopping_processes = []
        self.stop_timer.stop()
        self._set_busy(False)

    def stop_job(self) -> None:
        if self.action is None or self.stopping or self.process.state() != QProcess.ProcessState.Running:
            return
        self.stopping = True
        self._set_busy(True)
        self.status.setText("Stopping the study…")
        # The Windows virtual environment launcher may have a separate Python child.
        try:
            parent = psutil.Process(self.process.processId())
            self.stopping_processes = list(reversed(parent.children(recursive=True)))
            for child in self.stopping_processes:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            self.stopping = False
            self._set_busy(True)
            self.status.setText(f"Could not stop the study worker. {exc}")
            return
        self.process.kill()
        self.stop_timer.start()

    def _check_stopped(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        for child in self.stopping_processes:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    return
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                if child.is_running():
                    return
        if self.response_path.is_file():
            try:
                response = json.loads(self.response_path.read_text(encoding="utf-8"))
                expected_status = "mesh_ready" if self.action == "mesh" else "completed_unverified"
                if response.get("action") == self.action and response.get("status") == expected_status:
                    self._show_result(expected_status, response["summary"])
                    self._finish_job()
                    return
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
        self.status.setText("Study cancelled. The submitted configuration and any partial output were kept.")
        self._finish_job()

    def _job_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.action is None:
            return
        self._read_log(final=True)
        if self.stopping:
            return
        try:
            response = json.loads(self.response_path.read_text(encoding="utf-8"))
            if response.get("action") != self.action:
                raise ValueError("The worker response does not match this action.")
            if response.get("status") == "failed":
                raise ValueError(response.get("error", "The study failed."))
            expected_status = "mesh_ready" if self.action == "mesh" else "completed_unverified"
            if (exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0
                    or response.get("status") != expected_status):
                raise ValueError(f"The worker did not report {expected_status} (exit code {exit_code}).")
            self._show_result(expected_status, response["summary"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.status.setText(f"Study could not complete. {exc}")
        finally:
            self._finish_job()

    def _show_result(self, status: str, summary: dict) -> None:
        directory = Path(summary["output_directory"])
        if not directory.is_dir():
            raise ValueError("The result folder is missing.")
        if status == "mesh_ready":
            faces = summary["end_face_dofs"]
            message = (f"Mesh ready · {summary['ndof']:,} body DOFs and {faces['xleft']:,}/{faces['xright']:,} end-face DOFs. "
                       "Mechanical assembly and solving have not run.")
            if not summary["within_body_budget"]:
                message += f" Body DOFs exceed the current cap of {summary['max_dofs']:,}."
        else:
            report = directory / "report_AP_pnc.html"
            if not report.is_file():
                raise ValueError("The study completed but its report is missing.")
            message = "Computed (unverified). Mesh and physical convergence require separate checks."
            if summary.get("analysis_kind") == "scattering":
                power = "passed" if summary["power_balance_passed"] else "failed"
                message += f" Power-balance check {power}."
            self.report_path = report
        self.result_directory = directory
        self.report_button.setEnabled(self.report_path is not None)
        self.folder_button.setEnabled(True)
        self.status.setText(message)

    def _open(self, path: Path | None) -> None:
        if path is not None and (not path.exists() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))):
            self.status.setText(f"Could not open {path}")

    def closeEvent(self, event) -> None:
        if self.action is not None:
            self.status.setText("A study is still active. Use Stop or wait for the worker before closing.")
            event.ignore()
        else:
            event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = Workbench()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
