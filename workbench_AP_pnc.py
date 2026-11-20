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

from tools import browse_device_results_AP_pnc as device_results

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
    ("26-hole device · regional layers (coarse)", "historical_26hole_regional_AP_pnc.json"),
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
MAP_ACTIONS = ("map", "map-seed", "map-resume")
JOB_RESULTS = {"mesh": "mesh_ready", "run": "completed_unverified", "compare": "compared",
               **dict.fromkeys(RESUMABLE_ACTIONS, "completed_unverified"), **dict.fromkeys(MAP_ACTIONS, "partial")}
JOB_MESSAGES = {"mesh": "Checking the mesh…", "run": "Running the study…",
                "compare": "Comparing saved scattering runs…",
                "run-resumable": "Starting resumable ILU-LGMRES scattering…",
                "resume": "Resuming scattering from saved inputs…",
                "map": "Preparing the bounded frequency map…",
                "map-seed": "Checking saved mesh and operators…",
                "map-resume": "Resuming the frequency map from its saved inputs…"}


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
        self.report_title = None
        self.scattering_directory = None
        # These folders seed the explicit chooser dialogs.
        self.resume_directory = None
        self.resume_source = None
        self.map_directory = None
        self.map_source = None
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

        configuration = QGroupBox("Configure study")
        choices = QHBoxLayout(configuration)
        self.preset = QComboBox()
        self.preset.setAccessibleName("Simulation preset")
        for label, _ in PRESETS:
            self.preset.addItem(label)
        self.preset.addItem("Loaded configuration")
        self.load_button = QPushButton("Load configuration…")
        self.load_button.setToolTip("Load editable study inputs from a JSON file.")
        self.save_button = QPushButton("Save configuration…")
        choices.addWidget(QLabel("Preset"))
        choices.addWidget(self.preset, 1)
        choices.addWidget(self.load_button)
        choices.addWidget(self.save_button)
        main.addWidget(configuration)

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

        self.regional_controls = QGroupBox("Regional layered mesh")
        regional_layout = QHBoxLayout(self.regional_controls)
        self.regional_enabled = QCheckBox("Independent lead size")
        self.regional_lead_size = QLineEdit()
        self.regional_lead_size.setAccessibleName("Lead in-plane size (nm)")
        self.regional_lead_size.setMaximumWidth(100)
        self.regional_interval = QLineEdit()
        self.regional_interval.setAccessibleName("Central region interval (nm)")
        self.regional_interval.setPlaceholderText("start, end")
        self.regional_interval.setToolTip("Both cuts must lie in solid gaps outside all holes. The central region uses the main in-plane size.")
        self.regional_central_edge = QLineEdit()
        self.regional_central_edge.setAccessibleName("Central edge target (nm)")
        self.regional_central_edge.setPlaceholderText("Optional")
        self.regional_central_edge.setToolTip("Refine the central footprint after meshing. Leave blank or enter 0 to disable.")
        self.regional_central_edge.setMaximumWidth(100)
        self.regional_central_edge.setEnabled(False)
        central_edge_label = QLabel("Central edge target (nm)")
        central_edge_label.setBuddy(self.regional_central_edge)
        regional_layout.addWidget(self.regional_enabled)
        regional_layout.addWidget(QLabel("Lead size (nm)"))
        regional_layout.addWidget(self.regional_lead_size)
        regional_layout.addWidget(QLabel("Central interval (nm)"))
        regional_layout.addWidget(self.regional_interval, 1)
        regional_layout.addWidget(central_edge_label)
        regional_layout.addWidget(self.regional_central_edge)
        self.regional_enabled.toggled.connect(self._inputs_changed)
        self.regional_enabled.toggled.connect(self.regional_central_edge.setEnabled)
        self.regional_lead_size.textEdited.connect(self._inputs_changed)
        self.regional_interval.textEdited.connect(self._inputs_changed)
        self.regional_central_edge.textEdited.connect(self._inputs_changed)
        main.addWidget(self.regional_controls)

        resume_controls = QHBoxLayout()
        self.save_progress = QCheckBox("Save progress for resume (ILU-LGMRES)")
        self.save_progress.setToolTip("Use ILU-LGMRES and save incident-channel checkpoints for this run.")
        self.save_progress.toggled.connect(self._inputs_changed)
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

        self.map_controls = QGroupBox("Frequency map")
        map_layout = QHBoxLayout(self.map_controls)
        self.map_frequencies = QLineEdit()
        self.map_frequencies.setAccessibleName("Map frequencies (GHz)")
        self.map_frequencies.setPlaceholderText("Increasing samples, for example 0.25, 0.30, 0.35")
        self.map_frequencies.setToolTip("Explicit increasing samples. Missing frequencies are not interpolated.")
        self.map_max_points = QLineEdit("1")
        self.map_max_points.setAccessibleName("Maximum new point attempts")
        self.map_max_points.setMaximumWidth(55)
        self.map_max_points.setToolTip("Maximum point attempts in this invocation, including failed attempts.")
        self.map_button = QPushButton("Run frequency map")
        self.seed_map_button = QPushButton("Run from saved mesh…")
        self.seed_map_button.setToolTip("Use the selected native map’s geometry, material, mesh and operators "
                                        "with the samples and point limit entered here.")
        self.resume_map_button = QPushButton("Resume map…")
        self.resume_map_button.setToolTip("Use the saved configuration and frequency list; current study edits are ignored.")
        map_layout.addWidget(QLabel("Samples (GHz)"))
        map_layout.addWidget(self.map_frequencies, 1)
        map_layout.addWidget(QLabel("New point limit"))
        map_layout.addWidget(self.map_max_points)
        for button in (self.map_button, self.seed_map_button, self.resume_map_button):
            map_layout.addWidget(button)
        main.addWidget(self.map_controls)

        run_controls = QGroupBox("Run study")
        actions = QHBoxLayout(run_controls)
        self.mesh_button = QPushButton("Check mesh")
        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet("font-weight: 600; padding-left: 15px; padding-right: 15px;")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        for button in (self.mesh_button, self.run_button, self.stop_button):
            actions.addWidget(button)
        actions.addStretch()
        main.addWidget(run_controls)

        inspection = QGroupBox("Inspect and compare results")
        inspection_layout = QVBoxLayout(inspection)
        inspection_buttons = QHBoxLayout()
        self.saved_run_button = QPushButton("Load saved study…")
        self.saved_run_button.setToolTip("Load editable inputs and set their saved result as the current report.")
        self.saved_map_button = QPushButton("Open saved map…")
        self.saved_map_button.setToolTip("Open a saved frequency map in the browser and set it as the current report. Study edits are kept.")
        self.device_results_button = QPushButton("Browse device results")
        self.device_results_button.setToolTip("Open the read-only device dataset in a separate browser view.")
        self.compare_button = QPushButton("Compare with saved run…")
        self.report_button = QPushButton("Open report")
        self.folder_button = QPushButton("Open folder")
        for button in (self.saved_run_button, self.saved_map_button, self.device_results_button, self.compare_button,
                       self.report_button, self.folder_button):
            inspection_buttons.addWidget(button)
        inspection_buttons.addStretch()
        inspection_layout.addLayout(inspection_buttons)
        self.current_report = QLabel()
        self.current_report.setTextFormat(Qt.TextFormat.PlainText)
        self.current_report.setWordWrap(True)
        self.current_report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inspection_layout.addWidget(self.current_report)
        main.addWidget(inspection)
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
        self.device_results_button.clicked.connect(self.open_device_results)
        self.save_button.clicked.connect(self._choose_save)
        self.mesh_button.clicked.connect(lambda: self.start_job("mesh"))
        self.run_button.clicked.connect(lambda: self.start_job("run-resumable" if self.save_progress.isChecked() else "run"))
        self.resume_button.clicked.connect(self._choose_resume)
        self.map_button.clicked.connect(lambda: self.start_job("map"))
        self.seed_map_button.clicked.connect(self._choose_seed_map)
        self.resume_map_button.clicked.connect(self._choose_resume_map)
        self.saved_map_button.clicked.connect(self._choose_saved_map)
        self.map_frequencies.textEdited.connect(self._inputs_changed)
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
        path = QFileDialog.getExistingDirectory(self, "Load a saved study and its result", str(self.output_root))
        if path:
            self.open_saved_run(Path(path))

    def _choose_saved_map(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open a saved frequency map", str(self.map_directory or self.output_root))
        if directory:
            self.open_saved_map(Path(directory))

    def open_saved_map(self, directory: Path) -> bool:
        if self.action is not None:
            return False
        try:
            from tools import map_scattering_AP_pnc as mapper

            directory = directory.resolve()
            result = mapper.read_saved_map(directory)
            self._clear_result()
            self._show_map_result({**result, "output_directory": str(directory),
                                   "report_path": str(directory / mapper.REPORT)}, saved=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText(f"Could not open saved map. {exc}")
            return False
        return self._open(self.report_path)

    def open_device_results(self) -> bool:
        if self.action is not None:
            return False
        try:
            dataset = device_results.load_dataset()
            publication = device_results.publish_dataset(dataset, self.output_root / "saved_device_results_AP_pnc")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText(f"Could not open saved device results. {exc}")
            return False
        if not self._open(Path(publication["report_path"])):
            return False
        self.status.setText("Opened the read-only device dataset and its exports in the browser. "
                            "Spatial and frequency convergence remain unresolved.")
        return True

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
        self.status.setText(f"Loaded saved study. {self.status.text()}")
        return True

    def _apply_configuration(self, config: dict, path: Path, index: int) -> None:
        self.config = config
        was_blocked = self.preset.blockSignals(True)
        self.preset.setCurrentIndex(index)
        self.preset.blockSignals(was_blocked)
        mesh = config["mesh"]
        analysis = config["analysis"]
        layered = mesh.get("kind") == "layered"
        self.regional_controls.setVisible(layered)
        self.regional_enabled.setChecked("lead_mesh_size_nm" in mesh or "patterned_x_interval_nm" in mesh)
        self.regional_lead_size.setText(str(mesh.get("lead_mesh_size_nm", "")))
        interval = mesh.get("patterned_x_interval_nm", "")
        self.regional_interval.setText(", ".join(map(str, interval)) if isinstance(interval, (list, tuple)) else str(interval))
        self.regional_central_edge.setText(str(mesh.get("central_max_edge_nm", "")))
        gmsh = mesh.get("kind", "structured") in ("gmsh", "layered")
        scattering = analysis["kind"] == "scattering"
        self.save_progress.setChecked(False)
        self.save_progress.setVisible(scattering)
        self.map_controls.setVisible(scattering)
        self.map_frequencies.setText(str(analysis.get("frequency_GHz", "")))
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
            if "lead_mesh_size_nm" in mesh:
                lines.append("Regional mesh: lead and central sizes are independent; cut planes share a conforming trace.")
            if "central_max_edge_nm" in mesh:
                lines.append(f"Central footprint refinement edge target: {mesh['central_max_edge_nm']} nm")
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
        if config["mesh"].get("kind") == "layered":
            if self.regional_enabled.isChecked():
                try:
                    size = float(self.regional_lead_size.text().strip())
                    interval = [float(value) for value in re.split(r"[,\s]+", self.regional_interval.text().strip())]
                    valid = (math.isfinite(size) and size > 0 and len(interval) == 2
                             and all(math.isfinite(value) for value in interval)
                             and 0 < interval[0] < interval[1] < config["geometry"]["length_nm"])
                except (ValueError, OverflowError):
                    valid = False
                if not valid:
                    raise ValueError("Regional mesh needs a positive lead size and 0 < start < end < beam length, in nm.")
                original_size = config["mesh"].get("lead_mesh_size_nm")
                config["mesh"]["lead_mesh_size_nm"] = (
                    original_size if type(original_size) in (int, float) and original_size == size else size)
                original_interval = config["mesh"].get("patterned_x_interval_nm")
                config["mesh"]["patterned_x_interval_nm"] = (
                    original_interval if isinstance(original_interval, (list, tuple))
                    and len(original_interval) == 2
                    and all(type(value) in (int, float) for value in original_interval)
                    and list(original_interval) == interval else interval)
                text = self.regional_central_edge.text().strip()
                try:
                    target = float(text) if text else 0
                except (ValueError, OverflowError):
                    target = math.nan
                if not math.isfinite(target) or target < 0:
                    raise ValueError("The central edge target (nm) must be positive, blank, or 0.")
                if target == 0:
                    config["mesh"].pop("central_max_edge_nm", None)
                else:
                    original_target = config["mesh"].get("central_max_edge_nm")
                    config["mesh"]["central_max_edge_nm"] = (
                        original_target if type(original_target) in (int, float) and original_target == target else target)
            else:
                config["mesh"].pop("lead_mesh_size_nm", None)
                config["mesh"].pop("patterned_x_interval_nm", None)
                config["mesh"].pop("central_max_edge_nm", None)
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
        self.report_title = None
        self.scattering_directory = None
        self._set_result_buttons(False)

    def _set_result_buttons(self, busy: bool) -> None:
        self.compare_button.setEnabled(not busy and self.scattering_directory is not None)
        self.report_button.setEnabled(not busy and self.report_path is not None)
        self.folder_button.setEnabled(not busy and self.result_directory is not None)
        self.current_report.setText("Current report · " + (self.report_title or self.report_path.parent.name)
                                    if self.report_path is not None else "No current report")
        path = str(self.report_path) if self.report_path is not None else ""
        self.current_report.setToolTip(path)
        self.report_button.setToolTip(path)
        self.compare_button.setToolTip(f"Current comparison candidate: {self.scattering_directory}"
                                       if self.scattering_directory is not None else "")

    def _inputs_changed(self) -> None:
        self._clear_result()
        self.status.setText("Inputs changed. Check the mesh or run the study.")

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.preset, self.load_button, self.saved_run_button, self.saved_map_button,
                       self.device_results_button, self.save_button, self.form,
                       self.mesh_button, self.run_button, self.save_progress, self.wall_seconds, self.resume_button,
                       self.map_controls, self.regional_controls):
            widget.setEnabled(not busy)
        mapping = self.action in MAP_ACTIONS
        self.stop_button.setText("Stop after current point" if mapping else "Stop")
        self.stop_button.setToolTip("Request a stop at the next point boundary. Accepted points stay saved." if mapping else "")
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

    def _choose_resume_map(self) -> None:
        if self.action is not None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose a native frequency-map checkpoint",
                                                     str(self.map_directory or self.output_root))
        if directory:
            self.start_job("map-resume", map_directory=Path(directory))

    def _choose_seed_map(self) -> None:
        if self.action is not None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose the saved map whose geometry and mesh to use",
                                                     str(self.map_directory or self.output_root))
        if directory:
            self.start_job("map-seed", donor_map=Path(directory))

    def _frequency_list(self) -> list[float]:
        try:
            frequencies = [float(value) for value in re.split(r"[,\s]+", self.map_frequencies.text().strip())]
        except ValueError:
            frequencies = []
        if (not frequencies or any(not math.isfinite(value * 1e9) or value <= 0 for value in frequencies)
                or any(right <= left for left, right in zip(frequencies, frequencies[1:]))):
            raise ValueError("Map frequencies must be a strictly increasing list of positive numbers in GHz.")
        return frequencies

    def _map_point_bound(self) -> int:
        try:
            bound = int(self.map_max_points.text().strip())
        except ValueError:
            bound = 0
        if bound < 1:
            raise ValueError("Maximum new point attempts must be a positive whole number.")
        return bound

    def _wall_time(self) -> float:
        error = "Wall-time limit (s) must be a finite positive number."
        try:
            seconds = float(self.wall_seconds.text())
        except ValueError:
            raise ValueError(error) from None
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(error)
        return seconds

    def start_job(self, action: str, reference: Path | None = None, *, resume_from: Path | None = None,
                  map_directory: Path | None = None, donor_map: Path | None = None) -> bool:
        if self.action is not None or action not in JOB_RESULTS:
            return False
        comparing = action == "compare"
        resuming = action == "resume"
        resumable = action in RESUMABLE_ACTIONS
        mapping = action in MAP_ACTIONS
        if comparing and (reference is None or self.scattering_directory is None):
            return False
        if resuming and resume_from is None:
            return False
        if action == "map-resume" and map_directory is None:
            return False
        if action == "map-seed" and donor_map is None:
            return False
        if not comparing:
            self._clear_result()
        try:
            wall_seconds = self._wall_time() if resumable else None
            point_bound = self._map_point_bound() if mapping else None
            if comparing:
                reference = reference.resolve()
                if reference == self.scattering_directory:
                    raise ValueError("Choose a reference run different from the current result.")
            elif action == "map-resume":
                map_directory = map_directory.resolve()
                if not map_directory.is_dir():
                    raise ValueError("Choose a saved native frequency-map folder.")
            elif action == "map-seed":
                donor_map = donor_map.resolve()
                if not donor_map.is_dir():
                    raise ValueError("Choose a saved native frequency-map folder with mesh and operators.")
                frequencies = self._frequency_list()
            elif resuming:
                resume_from = resume_from.resolve()
                if not resume_from.is_dir():
                    raise ValueError("Choose a saved scattering attempt folder.")
            else:
                config = self.edited_config()
                if mapping:
                    if config["analysis"]["kind"] != "scattering":
                        raise ValueError("Frequency mapping requires a matched-port scattering configuration.")
                    frequencies = self._frequency_list()
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
            elif action == "map-resume":
                self._write_config(directory / "map_resume_request_AP_pnc.json", {
                    "author": "Athul Prem", "file_tag": "_AP_pnc", "action": action,
                    "map_directory": str(map_directory), "max_new_points": point_bound,
                })
                job_args = []
            elif action == "map-seed":
                map_directory = directory / "map_AP_pnc"
                self._write_config(directory / "map_seed_request_AP_pnc.json", {
                    "author": "Athul Prem", "file_tag": "_AP_pnc", "action": action,
                    "from_map": str(donor_map), "map_directory": str(map_directory),
                    "frequencies_GHz": frequencies, "max_new_points": point_bound,
                })
                job_args = ["--from-map", str(donor_map), "--frequencies-ghz", *(str(value) for value in frequencies)]
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
                if mapping:
                    map_directory = directory / "map_AP_pnc"
                    job_args.extend(("--frequencies-ghz", *(str(value) for value in frequencies)))
            if resumable:
                job_args.extend(("--wall-seconds", str(wall_seconds)))
            if mapping:
                job_args.extend(("--map-directory", str(map_directory), "--max-new-points", str(point_bound)))
        except (OSError, ValueError) as exc:
            task = "comparison" if comparing else "frequency map" if mapping else "study"
            self.status.setText(f"Could not start the {task}. {exc}")
            return False
        self.response_path = directory / "response_AP_pnc.json"
        self.log_path = directory / "worker_AP_pnc.log"
        self.resume_source = resume_from if resuming else None
        self.map_source = map_directory if mapping else None
        self.action = action
        self.log.clear()
        self.log_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._set_busy(True)
        self.status.setText(f"Using the geometry and preparation from {donor_map.name}…" if action == "map-seed"
                            else JOB_MESSAGES[action])
        if resumable or mapping:
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
        self.map_source = None
        self.stop_timer.stop()
        self.progress_timer.stop()
        self._set_busy(False)

    def stop_job(self) -> None:
        if self.action is None or self.stopping or self.process.state() != QProcess.ProcessState.Running:
            return
        self.stopping = True
        self._set_busy(True)
        if self.action in MAP_ACTIONS:
            try:
                self.response_path.with_suffix(".cancel").touch()
            except OSError as exc:
                self.stopping = False
                self._set_busy(True)
                self.status.setText(f"Could not request a stop after the current point. {exc}")
                return
            self.status.setText("Stop requested at a point boundary. An active point can finish. Accepted points stay saved.")
            self.stop_deadline = None
            self.stop_timer.start()
            return
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
            if not self._show_map_fallback("Frequency map stopped.") and not self._show_resume_fallback("Scattering stopped."):
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
        if self.action in MAP_ACTIONS:
            preparation = ("Restoring copied mesh and operators…" if self.action == "map-seed"
                           else "Preparing shared mesh and operators…")
            phase = {"seeding": "Verifying and copying saved mesh and operators…", "preparing": preparation,
                     "solving": "Solving frequency point…",
                     "point_finished": "Frequency point finished.", "publishing": "Saving frequency map…"}.get(progress.get("phase"))
            if phase:
                count = f"Accepted frequencies: {progress.get('accepted_points', 0)}/{progress.get('total_points', '?')}."
                frequency = progress.get("current_frequency_GHz")
                detail = f" {frequency:g} GHz." if type(frequency) in (int, float) else ""
                self.status.setText(f"{phase}{detail} {count}")
            return
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
        if self.action in MAP_ACTIONS and status in ("complete", "partial") and isinstance(summary, dict):
            if not worker_succeeded and not summary.get("failed_points"):
                raise ValueError("The frequency-map worker did not exit successfully.")
            self._show_map_result(summary)
            return
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
            task = "Comparison" if self.action == "compare" else "Frequency map" if self.action in MAP_ACTIONS else "Study"
            failure = f"{task} could not complete. {exc}"
            if not self._show_map_fallback(failure) and not self._show_resume_fallback(failure):
                self.status.setText(failure)
        finally:
            self._finish_job()

    def _show_map_result(self, summary: dict, *, saved: bool = False) -> None:
        directory = Path(summary["output_directory"])
        if not directory.is_dir():
            raise ValueError("The frequency-map folder is missing.")
        points = summary["points"]
        accepted = sum(point["status"] == "accepted" for point in points)
        failed = sum(point["status"] == "failed" for point in points)
        if summary["status"] == "complete" and accepted != len(points):
            raise ValueError("The map claims completion with unaccepted points.")
        self.result_directory = self.map_directory = directory
        report = summary.get("report_path")
        self.report_path = Path(report) if report and Path(report).is_file() else None
        config = summary.get("inputs", {}).get("configuration", {})
        identity = f"{config['name']} · {directory.name}" if config.get("name") else directory.name
        self.report_title = f"{'Saved frequency map' if saved else 'Frequency map'} · {identity}"
        self.scattering_directory = None
        heading = "Frequency map complete" if summary["status"] == "complete" else "Partial frequency map"
        message = f"{heading} · {accepted}/{len(points)} accepted frequencies, {failed} failed."
        qualification = [point.get("qualification", {}).get("status") for point in points if point["status"] == "accepted"]
        passed_checks = qualification.count("finite_checks_passed")
        failed_checks = qualification.count("finite_checks_failed")
        if passed_checks or failed_checks:
            message += f" Recorded finite checks: {passed_checks} passed, {failed_checks} failed."
        if "preparation_reuse" in summary:
            geometry = config["geometry"]
            dimensions = " × ".join(f"{geometry[key]:g}" for key in ("length_nm", "width_nm", "thickness_nm"))
            message += f" Using saved geometry and operators for {config['name']} ({dimensions} nm)."
        if summary.get("stop_requested"):
            message += " Stopped at a point boundary."
        if summary["status"] != "complete" and summary.get("execution_mode") is None:
            message += " Use Resume map… with this folder to continue its saved inputs."
        message += " Spatial and frequency convergence remain unestablished."
        self.status.setText(message)
        self._set_result_buttons(self.action is not None)

    def _show_map_fallback(self, heading: str) -> bool:
        if self.action not in MAP_ACTIONS or self.map_source is None or not self.map_source.is_dir():
            return False
        self.result_directory = self.map_directory = self.map_source
        self.status.setText(f"{heading} Saved inputs and attempts are in {self.map_source}. "
                            "Use Resume map… to check the saved checkpoint and continue.")
        self._set_result_buttons(self.action is not None)
        return True

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
            reference = summary["reference"].get("directory")
            candidate = summary["candidate"].get("directory")
            comparison = (f"reference {Path(reference).name}, candidate {Path(candidate).name}"
                          if reference and candidate else directory.name)
            self.report_title = f"{heading} · {comparison}"
        else:
            report = directory / "report_AP_pnc.html"
            if not report.is_file():
                raise ValueError("The study completed but its report is missing.")
            message = "Computed (unverified). Mesh and physical convergence require separate checks."
            if summary.get("analysis_kind") == "scattering":
                power = "passed" if summary["power_balance_passed"] else "failed"
                message += f" Power-balance check {power}."
                qualification = summary.get("qualification", {}).get("status")
                if qualification == "finite_checks_passed":
                    message += " Finite numerical checks passed."
                elif qualification == "finite_checks_failed":
                    message += " Finite numerical checks failed."
                self.scattering_directory = directory.resolve()
            if self.action == "resume":
                message += " Resumed from the saved simulation inputs."
            self.report_path = report
            name = summary.get("configuration", {}).get("name")
            self.report_title = f"Study result · {name} · {directory.name}" if name else f"Study result · {directory.name}"
        self.result_directory = directory
        self._set_result_buttons(self.action is not None)
        self.status.setText(message)

    def _open(self, path: Path | None) -> bool:
        if path is None:
            return False
        if not path.exists() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.status.setText(f"Could not open {path}")
            return False
        return True

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
