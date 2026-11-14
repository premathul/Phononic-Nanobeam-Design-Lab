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
import time

import psutil

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
    from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QTextCursor
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
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
    ("Two-hole beam · layered mesh", "perforated_layered_AP_pnc.json"),
    ("Two-hole refined layers · scattering", "perforated_layered_reference_AP_pnc.json"),
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
    ("thickness_layers", "Thickness layers"),
    ("max_dofs", "Body DOF cap"), ("max_port_dofs", "Port DOF cap"),
)
NUMERICAL_FIELDS = {
    "geometry": ("length_nm", "width_nm", "thickness_nm"),
    "analysis": ("frequency_GHz", "num_modes", "k_points"),
    "mesh": ("mesh_size_nm", "port_size_nm", "circle_resolution", "thickness_layers"),
    "resources": ("max_dofs", "max_port_dofs"),
}
WHOLE_FIELDS = {"num_modes", "k_points", "circle_resolution", "thickness_layers", "max_dofs", "max_port_dofs"}
RESUMABLE_ACTIONS = ("run-resumable", "resume")
JOB_RESULTS = {"mesh": "mesh_ready", "run": "completed_unverified", "compare": "compared",
               **dict.fromkeys(RESUMABLE_ACTIONS, "completed_unverified")}
JOB_MESSAGES = {"mesh": "Checking the mesh…", "run": "Running the study…",
                "compare": "Comparing saved scattering runs…",
                "run-resumable": "Starting resumable ILU-LGMRES scattering…",
                "resume": "Resuming scattering from saved inputs…"}


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
    if config["mesh"].get("kind", "structured") not in ("structured", "gmsh", "layered"):
        raise ValueError("The mesh kind must be structured, gmsh, or layered.")
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
        self.scattering_directory = None
        self.resume_directory = None
        self.resume_source = None
        self.action = None
        self.response_path = None
        self.stopping = False
        self.stopping_processes = []
        self.stop_deadline = None
        self.stop_timer = QTimer(self)
        self.stop_timer.setInterval(100)
        self.stop_timer.timeout.connect(self._check_stopped)
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(300)
        self.progress_timer.timeout.connect(self._update_progress)
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
        self.saved_run_button = QPushButton("Open saved run…")
        self.save_button = QPushButton("Save JSON…")
        choices.addWidget(QLabel("Preset"))
        choices.addWidget(self.preset, 1)
        choices.addWidget(self.load_button)
        choices.addWidget(self.save_button)
        choices.addWidget(self.saved_run_button)
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

        resume_controls = QHBoxLayout()
        self.save_progress = QCheckBox("Save progress for resume (ILU-LGMRES)")
        self.save_progress.setToolTip("Use ILU-LGMRES and save incident-channel checkpoints for this run.")
        self.wall_seconds = QLineEdit("1800")
        self.wall_seconds.setAccessibleName("Wall-time limit (s)")
        self.wall_seconds.setToolTip("Elapsed time budget in seconds for a resumable run or continuation.")
        self.wall_seconds.setMaximumWidth(90)
        wall_label = QLabel("Wall-time limit (s)")
        wall_label.setBuddy(self.wall_seconds)
        self.resume_button = QPushButton("Resume scattering…")
        self.resume_button.setToolTip("Select the saved attempt folder. Resume uses its original simulation inputs.")
        resume_controls.addWidget(self.save_progress)
        resume_controls.addStretch()
        resume_controls.addWidget(wall_label)
        resume_controls.addWidget(self.wall_seconds)
        resume_controls.addWidget(self.resume_button)
        main.addLayout(resume_controls)

        actions = QHBoxLayout()
        self.mesh_button = QPushButton("Check mesh")
        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet("font-weight: 600; padding-left: 15px; padding-right: 15px;")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.compare_button = QPushButton("Compare saved run…")
        self.report_button = QPushButton("Open report")
        self.folder_button = QPushButton("Open folder")
        for button in (self.mesh_button, self.run_button, self.stop_button, self.compare_button,
                       self.report_button, self.folder_button):
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
        self.saved_run_button.clicked.connect(self._choose_saved_run)
        self.save_button.clicked.connect(self._choose_save)
        self.mesh_button.clicked.connect(lambda: self.start_job("mesh"))
        self.run_button.clicked.connect(lambda: self.start_job("run-resumable" if self.save_progress.isChecked() else "run"))
        self.resume_button.clicked.connect(self._choose_resume)
        self.stop_button.clicked.connect(self.stop_job)
        self.compare_button.clicked.connect(self._choose_reference)
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

    def _choose_saved_run(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open a saved simulation run", str(self.output_root))
        if path:
            self.open_saved_run(Path(path))

    def load_configuration(self, path: Path, index: int = len(PRESETS)) -> bool:
        if self.action is not None:
            return False
        try:
            config = _read_configuration(path)
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not load configuration. {exc}")
            return False
        self._apply_configuration(config, path, index)
        self.status.setText("Configuration ready. Check the mesh or run the study.")
        return True

    def open_saved_run(self, directory: Path) -> bool:
        if self.action is not None:
            return False
        try:
            directory = directory.resolve()
            config_path = directory / "configuration_AP_pnc.json"
            config = _read_configuration(config_path)
            summary = json.loads((directory / "summary_AP_pnc.json").read_text(encoding="utf-8-sig"))
            if not isinstance(summary, dict) or summary.get("status") != "completed_unverified":
                raise ValueError("Choose a completed simulation run.")
            if summary.get("configuration") != config or summary.get("analysis_kind") != config["analysis"]["kind"]:
                raise ValueError("The saved configuration and result summary do not match.")
            if summary["analysis_kind"] == "scattering" and type(summary.get("power_balance_passed")) is not bool:
                raise ValueError("The saved scattering result is missing its power-balance check.")
            if not (directory / "report_AP_pnc.html").is_file():
                raise ValueError("The saved run report is missing.")
        except (OSError, ValueError) as exc:
            self.status.setText(f"Could not open saved run. {exc}")
            return False
        self._apply_configuration(config, config_path, len(PRESETS))
        self._show_result("completed_unverified", {**summary, "output_directory": str(directory)})
        return True

    def _apply_configuration(self, config: dict, path: Path, index: int) -> None:
        self.config = config
        was_blocked = self.preset.blockSignals(True)
        self.preset.setCurrentIndex(index)
        self.preset.blockSignals(was_blocked)
        mesh = config["mesh"]
        analysis = config["analysis"]
        layered = mesh.get("kind") == "layered"
        gmsh = mesh.get("kind", "structured") in ("gmsh", "layered")
        scattering = analysis["kind"] == "scattering"
        self.save_progress.setChecked(False)
        self.save_progress.setVisible(scattering)
        self.visible_fields = {"name", "length_nm", "width_nm", "thickness_nm", "max_dofs"}
        self.visible_fields.update(("mesh_size_nm", "port_size_nm", "circle_resolution") if gmsh else ("cells",))
        if layered:
            self.visible_fields.add("thickness_layers")
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
        for key, label in (("mesh_size_nm", "In-plane mesh size (nm)" if layered else "Body mesh size (nm)"),
                           ("port_size_nm", "Port width mesh size (nm)" if layered else "Port mesh size (nm)")):
            self.form_layout.labelForField(self.fields[key]).setText(label)
            self.fields[key].setAccessibleName(label)
        self._show_provenance(path)
        self.log.clear()
        self._clear_result()

    def _show_provenance(self, path: Path) -> None:
        material, analysis, mesh = (self.config[key] for key in ("material", "analysis", "mesh"))
        boundary = BOUNDARIES.get(analysis.get("boundary"), analysis.get("boundary", "unspecified"))
        lines = [f"Analysis: {ANALYSES[analysis['kind']]} · {boundary}"]
        if analysis["kind"] == "scattering":
            lines.append(f"Incident side: {analysis.get('incident_side', 'left')}")
        lines.append(f"Displacement element order: {mesh.get('element_order', 'unspecified')}")
        if mesh.get("kind") == "layered":
            lines.append("Layered 3D mesh: in-plane size and thickness layers are independent.")
            lines.append("Thickness layers also set the port thickness resolution.")
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
        self.scattering_directory = None
        self._set_result_buttons(False)

    def _set_result_buttons(self, busy: bool) -> None:
        self.compare_button.setEnabled(not busy and self.scattering_directory is not None)
        self.report_button.setEnabled(not busy and self.report_path is not None)
        self.folder_button.setEnabled(not busy and self.result_directory is not None)

    def _inputs_changed(self) -> None:
        self._clear_result()
        self.status.setText("Inputs changed. Check the mesh or run the study.")

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.preset, self.load_button, self.saved_run_button, self.save_button, self.form,
                       self.mesh_button, self.run_button, self.save_progress, self.wall_seconds, self.resume_button):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy and not self.stopping and self.process.state() == QProcess.ProcessState.Running)
        self._set_result_buttons(busy)

    def _choose_reference(self) -> None:
        if self.action is not None or self.scattering_directory is None:
            return
        reference = QFileDialog.getExistingDirectory(self, "Choose a saved reference scattering run", str(self.scattering_directory.parent))
        if reference:
            self.start_job("compare", Path(reference))

    def _choose_resume(self) -> None:
        if self.action is not None:
            return
        directory = self.resume_directory or self.output_root / "_checkpoints"
        if not directory.is_dir():
            directory = self.output_root
        attempt = QFileDialog.getExistingDirectory(self, "Choose a saved scattering attempt", str(directory))
        if attempt:
            self.start_job("resume", resume_from=Path(attempt))

    def _wall_time(self) -> float:
        error = "Wall-time limit (s) must be a finite positive number."
        try:
            seconds = float(self.wall_seconds.text())
        except ValueError:
            raise ValueError(error) from None
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(error)
        return seconds

    def start_job(self, action: str, reference: Path | None = None, *, resume_from: Path | None = None) -> bool:
        if self.action is not None or action not in JOB_RESULTS:
            return False
        comparing = action == "compare"
        resuming = action == "resume"
        resumable = action in RESUMABLE_ACTIONS
        if comparing and (reference is None or self.scattering_directory is None):
            return False
        if resuming and resume_from is None:
            return False
        if not comparing:
            self._clear_result()
        try:
            wall_seconds = self._wall_time() if resumable else None
            if comparing:
                reference = reference.resolve()
                if reference == self.scattering_directory:
                    raise ValueError("Choose a reference run different from the current result.")
            elif resuming:
                resume_from = resume_from.resolve()
                if not resume_from.is_dir():
                    raise ValueError("Choose a saved scattering attempt folder.")
            else:
                config = self.edited_config()
                if resumable:
                    if config["analysis"]["kind"] != "scattering":
                        raise ValueError("Saving progress requires a single-frequency scattering configuration.")
                    config["resources"]["linear_solver"] = "ilu_lgmres"
            jobs = self.output_root / "_workbench"
            jobs.mkdir(parents=True, exist_ok=True)
            directory = Path(mkdtemp(prefix="job-", dir=jobs))
            if comparing:
                candidate = self.scattering_directory
                self._write_config(directory / "comparison_request_AP_pnc.json", {
                    "author": "Athul Prem", "file_tag": "_AP_pnc", "action": action,
                    "reference": str(reference), "candidate": str(candidate),
                })
                job_args = ["--reference", str(reference), "--candidate", str(candidate)]
            elif resuming:
                self._write_config(directory / "resume_request_AP_pnc.json", {
                    "author": "Athul Prem", "file_tag": "_AP_pnc", "action": action,
                    "resume_from": str(resume_from), "wall_seconds": wall_seconds,
                })
                job_args = ["--resume-from", str(resume_from)]
            else:
                config_path = directory / "configuration_AP_pnc.json"
                self._write_config(config_path, config)
                job_args = ["--config", str(config_path)]
            if resumable:
                job_args.extend(("--wall-seconds", str(wall_seconds)))
        except (OSError, ValueError) as exc:
            task = "comparison" if comparing else "study"
            self.status.setText(f"Could not start the {task}. {exc}")
            return False
        self.response_path = directory / "response_AP_pnc.json"
        self.log_path = directory / "worker_AP_pnc.log"
        self.resume_source = resume_from if resuming else None
        self.action = action
        self.log.clear()
        self.log_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._set_busy(True)
        self.status.setText(JOB_MESSAGES[action])
        if resumable:
            self.progress_timer.start()
        self.process.start(sys.executable, ["-u", "-B", str(WORKER), *job_args,
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
            self.status.setText(f"Could not start the worker. {self.process.errorString()}")
            self._finish_job()

    def _finish_job(self) -> None:
        self.action = None
        self.stopping = False
        self.stopping_processes = []
        self.stop_deadline = None
        self.resume_source = None
        self.stop_timer.stop()
        self.progress_timer.stop()
        self._set_busy(False)

    def stop_job(self) -> None:
        if self.action is None or self.stopping or self.process.state() != QProcess.ProcessState.Running:
            return
        self.stopping = True
        self._set_busy(True)
        if self.action in RESUMABLE_ACTIONS:
            try:
                self.response_path.with_suffix(".cancel").touch()
            except OSError as exc:
                self.log.appendPlainText(f"Could not request a checkpointed stop. {exc}")
            else:
                self.status.setText("Stopping scattering and keeping saved checkpoints…")
                self.stop_deadline = time.monotonic() + 6
                self.stop_timer.start()
                return
        self._kill_worker()

    def _kill_worker(self) -> None:
        self.stop_deadline = None
        self.status.setText("Stopping the worker…")
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
            self.stop_timer.stop()
            self._set_busy(True)
            self.status.setText(f"Could not stop the worker. {exc}")
            return
        self.process.kill()
        self.stop_timer.start()

    def _check_stopped(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            if self.stop_deadline is not None and time.monotonic() >= self.stop_deadline:
                self._kill_worker()
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
        try:
            self._show_job_response(self._read_response(self.response_path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not self._show_resume_fallback("Scattering stopped."):
                task = "Comparison" if self.action == "compare" else "Study"
                self.status.setText(f"{task} cancelled. The submitted inputs and any partial output were kept.")
        self._finish_job()

    def _read_response(self, path: Path) -> dict:
        response = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(response, dict) or response.get("action") != self.action:
            raise ValueError("The worker response does not match this action.")
        return response

    def _read_progress(self) -> dict:
        try:
            response = self._read_response(self.response_path.with_suffix(".progress.json"))
            if response.get("status") == "running" and isinstance(response.get("summary"), dict):
                return response["summary"]
        except (OSError, ValueError):
            pass
        return {}

    @staticmethod
    def _accepted_channels(summary: dict, *, last_reported: bool = False) -> str:
        accepted = summary.get("accepted_channels")
        total = summary.get("total_channels")
        if type(accepted) is not int:
            return ""
        count = str(accepted) if type(total) is not int else f"{accepted}/{total}"
        label = "Last reported accepted incident channels" if last_reported else "Accepted incident channels"
        return f"{label}: {count}."

    def _update_progress(self) -> None:
        if self.stopping:
            return
        progress = self._read_progress()
        phase = {"staging": "Preparing mesh, operator and ports…", "solving": "Solving incident channels…",
                 "exporting": "Writing the scattering result…"}.get(progress.get("phase"))
        if phase:
            self.status.setText(f"{phase} {self._accepted_channels(progress)}".strip())

    def _show_resume_status(self, heading: str, summary: dict, *, last_reported: bool = False) -> None:
        parts = [heading, summary.get("message", ""), self._accepted_channels(summary, last_reported=last_reported)]
        path = summary.get("resume_directory")
        if path and Path(path).is_dir():
            self.result_directory = Path(path)
            if summary.get("can_resume"):
                self.resume_directory = self.result_directory
                parts.append("Use Resume scattering… and select the saved attempt folder.")
            parts.append(str(self.result_directory))
        self._set_result_buttons(self.action is not None)
        self.status.setText(" ".join(part for part in parts if part))

    def _show_resume_fallback(self, heading: str) -> bool:
        if self.action not in RESUMABLE_ACTIONS:
            return False
        progress = self._read_progress()
        path = progress.get("resume_directory") or self.resume_source
        if not path or not Path(path).is_dir():
            return False
        self._show_resume_status(heading, {**progress, "resume_directory": str(path), "can_resume": True,
                                           "message": "The worker did not write its final status. "
                                                      "Resume checks the saved checkpoints."}, last_reported=True)
        return True

    def _show_job_response(self, response: dict, worker_succeeded: bool = True) -> None:
        status, summary = response.get("status"), response.get("summary")
        if self.action in RESUMABLE_ACTIONS and isinstance(summary, dict):
            heading = {"interrupted": "Scattering interrupted.", "blocked": "Scattering blocked.",
                       "failed": "Scattering failed."}.get(status)
            if heading:
                self._show_resume_status(heading, summary)
                return
        if status == "failed":
            raise ValueError(response.get("error", "The worker failed."))
        expected = JOB_RESULTS[self.action]
        if status != expected or not worker_succeeded:
            raise ValueError(f"The worker did not report {expected}.")
        self._show_result(expected, summary)

    def _job_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.action is None:
            return
        self._read_log(final=True)
        if self.stopping:
            return
        try:
            response = self._read_response(self.response_path)
            self._show_job_response(response, exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            task = "Comparison" if self.action == "compare" else "Study"
            failure = f"{task} could not complete. {exc}"
            if not self._show_resume_fallback(failure):
                self.status.setText(failure)
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
        elif status == "compared":
            report = Path(summary["report_path"])
            if not report.is_file():
                raise ValueError("The comparison report is missing.")
            power = "passed" if summary["power_passed"] else "failed"
            complex_response = "passed" if summary["complex_passed"] else "failed"
            reference_balance = "passed" if summary["reference"]["power_balance_passed"] else "failed"
            candidate_balance = "passed" if summary["candidate"]["power_balance_passed"] else "failed"
            repeatability = summary["comparison_kind"] == "repeatability"
            heading = "Repeatability" if repeatability else "Mesh sensitivity"
            message = (f"{heading} · maximum power change {summary['maximum_power_change']:.5g} ({power}). "
                       f"Maximum complex-response change {summary['maximum_complex_change']:.5g} ({complex_response}). "
                       f"Input power balance: reference {reference_balance}, candidate {candidate_balance}. ")
            if repeatability:
                message += "Both runs use the same discretization."
            else:
                acceptance = "passed" if summary["mesh_pair_passed"] else "failed"
                message += f"Mesh-pair check {acceptance} for these runs."
            self.report_path = report
        else:
            report = directory / "report_AP_pnc.html"
            if not report.is_file():
                raise ValueError("The study completed but its report is missing.")
            message = "Computed (unverified). Mesh and physical convergence require separate checks."
            if summary.get("analysis_kind") == "scattering":
                power = "passed" if summary["power_balance_passed"] else "failed"
                message += f" Power-balance check {power}."
                self.scattering_directory = directory.resolve()
            if self.action == "resume":
                message += " Resumed from the saved simulation inputs."
            self.report_path = report
        self.result_directory = directory
        self._set_result_buttons(self.action is not None)
        self.status.setText(message)

    def _open(self, path: Path | None) -> None:
        if path is not None and (not path.exists() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))):
            self.status.setText(f"Could not open {path}")

    def closeEvent(self, event) -> None:
        if self.action is not None:
            self.status.setText("A job is still active. Use Stop or wait for the worker before closing.")
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
