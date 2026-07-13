"""Audio device selector dialog using PySide6.

Returns an ``AudioEngineConfig`` for constructing an ``AudioEngine``.
Supports optional config-folder profiles (JSON).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .config import (
    AudioEngineConfig,
    Dtype,
    Latency,
    StreamConfig,
    list_profiles,
    profile_path,
)


def list_apis() -> list[tuple[int, str]]:
    """Return (id, name) for all host APIs."""
    return [(i, a["name"]) for i, a in enumerate(sd.query_hostapis())]


def list_channels(api_id: int) -> list[int]:
    """Return the distinct max_output_channels values available on a host API."""
    api = sd.query_hostapis(api_id)
    chans: set[int] = set()
    for i in api["devices"]:
        dev = sd.query_devices(i)
        if dev["max_output_channels"] > 0:
            chans.add(int(dev["max_output_channels"]))
    return sorted(chans)


def list_output_devices(api_id: int | None = None) -> list[tuple[int, str, int]]:
    """Return (index, name, max_output_channels) for output devices."""
    out: list[tuple[int, str, int]] = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            if api_id is not None and dev["hostapi"] != api_id:
                continue
            out.append((i, str(dev["name"]), int(dev["max_output_channels"])))
    return out


def test_silent_output(config: AudioEngineConfig, *, duration: float = 0.05) -> None:
    """Open the configured output briefly and write silence.

    Raises on PortAudio / device errors.
    """

    def callback(
        outdata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        outdata.fill(0)

    with sd.OutputStream(callback=callback, **config.stream_kwargs()):
        sd.sleep(int(duration * 1000))


def _clear_layout(layout: QHBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


class DeviceConfigSelector(QDialog):
    """Modal dialog that returns an ``AudioEngineConfig``."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        config_dir: str | Path | None = None,
        profile: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio Device Configuration")
        self._channel_values: list[int] = []
        self._config_dir: Path | None = (
            Path(config_dir) if config_dir is not None else None
        )
        self._saved_snapshot: dict[str, Any] | None = None
        self._loading = False
        self._requested_profile = profile

        self._build_ui()

        if self._config_dir is not None:
            self._refresh_profile_ui(selected=profile)
            self._apply_requested_profile(warn_if_missing=True)
        else:
            self._seed_form()
            self._mark_clean()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        profile_box = QGroupBox("Profile")
        profile_form = QFormLayout(profile_box)

        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("No config folder selected")
        profile_form.addRow("Folder:", self.folder_edit)

        folder_btns = QHBoxLayout()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_folder)
        new_folder_btn = QPushButton("New config folder…")
        new_folder_btn.clicked.connect(self._on_new_folder)
        folder_btns.addWidget(browse_btn)
        folder_btns.addWidget(new_folder_btn)
        folder_btns.addStretch()
        profile_form.addRow("", folder_btns)

        profile_row = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(False)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_index_changed)
        self.profile_name_edit = QLineEdit()
        self.profile_name_edit.setPlaceholderText("Profile name")
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(lambda: self._on_save())
        profile_row.addWidget(self.profile_combo, stretch=1)
        profile_row.addWidget(self.profile_name_edit, stretch=1)
        profile_row.addWidget(self.save_button)
        profile_form.addRow("Profile:", profile_row)

        self.profile_hint = QLabel(
            "Pick a profile from the list, or edit the name and Save to create a new one."
        )
        self.profile_hint.setStyleSheet("color: gray;")
        profile_form.addRow("", self.profile_hint)
        layout.addWidget(profile_box)

        form = QFormLayout()

        self.api_combo = QComboBox()
        for api_id, api_name in list_apis():
            self.api_combo.addItem(api_name, userData=api_id)
        self.api_combo.currentIndexChanged.connect(self._on_api_changed)
        form.addRow("API:", self.api_combo)

        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        form.addRow("Device:", self.device_combo)

        self.channels_group = QButtonGroup(self)
        self.channels_container = QWidget()
        self.channels_layout = QHBoxLayout(self.channels_container)
        self.channels_layout.setContentsMargins(0, 0, 0, 0)
        form.addRow("Channels:", self.channels_container)

        self.fs_combo = QComboBox()
        self.fs_combo.addItems(["44100", "48000", "96000"])
        form.addRow("Sample rate:", self.fs_combo)

        self.exclusive_check = QCheckBox("Exclusive mode")
        self.exclusive_check.setChecked(False)
        form.addRow("", self.exclusive_check)

        self.loaded_profile_label = QLabel("")
        self.loaded_profile_label.setStyleSheet("color: gray;")
        form.addRow("", self.loaded_profile_label)
        layout.addLayout(form)

        self.advanced_group = QGroupBox("Advanced")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_form = QFormLayout()

        self.blocksize_combo = QComboBox()
        self.blocksize_combo.addItem("Auto (0)", userData=0)
        for bs in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            self.blocksize_combo.addItem(str(bs), userData=bs)
        advanced_form.addRow("Blocksize:", self.blocksize_combo)

        self.latency_combo = QComboBox()
        self.latency_combo.addItem("low", userData="low")
        self.latency_combo.addItem("high", userData="high")
        self.latency_combo.setCurrentIndex(1)
        advanced_form.addRow("Latency:", self.latency_combo)

        self.dtype_combo = QComboBox()
        for dtype_name in ("float32", "int16", "int32", "uint8"):
            self.dtype_combo.addItem(dtype_name, userData=dtype_name)
        advanced_form.addRow("Data type:", self.dtype_combo)

        self.clip_off_check = QCheckBox("Disable clipping (clip_off)")
        advanced_form.addRow("", self.clip_off_check)

        self.dither_off_check = QCheckBox("Disable dither (dither_off)")
        advanced_form.addRow("", self.dither_off_check)

        self.never_drop_input_check = QCheckBox("Never drop input (never_drop_input)")
        advanced_form.addRow("", self.never_drop_input_check)

        self.prime_check = QCheckBox("Prime output buffers using stream callback")
        advanced_form.addRow("", self.prime_check)

        self.advanced_group.setLayout(advanced_form)
        layout.addWidget(self.advanced_group)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(self._on_test)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QPushButton("OK")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self._on_ok)
        button_row.addWidget(self.test_button)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.ok_button)
        layout.addLayout(button_row)

        self._connect_settings_reset()

    def _apply_requested_profile(self, *, warn_if_missing: bool) -> None:
        """Load ``profile=`` (or the combo selection) into the device form."""
        name = (self._requested_profile or self._current_profile_name() or "").strip()
        if name and self._load_profile(name):
            return
        if warn_if_missing and self._requested_profile:
            QMessageBox.warning(
                self,
                "Profile not found",
                f"No profile named “{self._requested_profile}” in:\n{self._config_dir}",
            )
        self.loaded_profile_label.setText("")
        self._seed_form()
        self._mark_clean()

    def _seed_form(self) -> None:
        """Pick a real host API + device so the form has concrete values to edit."""
        prefer = 0
        for i in range(self.api_combo.count()):
            data = self.api_combo.itemData(i)
            if data is not None and "WASAPI" in sd.query_hostapis(data)["name"].upper():
                prefer = i
                break
        if self.api_combo.currentIndex() == prefer:
            self._on_api_changed(prefer)
        else:
            self.api_combo.setCurrentIndex(prefer)

    # --- profile / folder -------------------------------------------------

    def _refresh_profile_ui(self, selected: str | None = None) -> None:
        if self._config_dir is not None:
            self.folder_edit.setText(str(self._config_dir.resolve()))
        else:
            self.folder_edit.clear()

        profiles = list_profiles(self._config_dir) if self._config_dir else []
        if "default" in profiles:
            profiles = ["default", *[p for p in profiles if p != "default"]]
        choose = (selected or self.profile_name_edit.text() or "").strip()
        if choose.lower().endswith(".json"):
            choose = choose[:-5]

        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for name in profiles:
            self.profile_combo.addItem(name)
        if choose and choose in profiles:
            self.profile_combo.setCurrentIndex(profiles.index(choose))
            self.profile_name_edit.setText(choose)
        elif selected:
            # Explicit request missing — do not silently fall through to another profile.
            self.profile_combo.setCurrentIndex(-1)
            self.profile_name_edit.setText(choose)
        elif profiles:
            self.profile_combo.setCurrentIndex(0)
            self.profile_name_edit.setText(profiles[0])
        else:
            self.profile_name_edit.setText(choose or "default")
        self.profile_combo.blockSignals(False)

        enabled = self._config_dir is not None
        self.profile_combo.setEnabled(enabled)
        self.profile_name_edit.setEnabled(enabled)
        self.save_button.setEnabled(enabled)

    def _set_profile_name(self, name: str) -> None:
        stem = name[:-5] if name.lower().endswith(".json") else name.strip()
        self.profile_name_edit.setText(stem)
        idx = self.profile_combo.findText(stem)
        self.profile_combo.blockSignals(True)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def _current_profile_name(self) -> str:
        return self.profile_name_edit.text().strip()

    def _on_profile_index_changed(self, index: int) -> None:
        if self._loading or self._config_dir is None or index < 0:
            return
        name = self.profile_combo.itemText(index).strip()
        if not name:
            return
        if self._is_dirty() and not self._confirm_discard_if_dirty():
            # Revert combo to the currently loaded profile name.
            self.profile_combo.blockSignals(True)
            prev = self._current_profile_name()
            idx = self.profile_combo.findText(prev)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
            self.profile_combo.blockSignals(False)
            return
        self._load_profile(name)

    def _mark_clean(self, snapshot: dict[str, Any] | None = None) -> None:
        self._saved_snapshot = (
            snapshot if snapshot is not None else self._collect_config().to_dict()
        )

    def _is_dirty(self) -> bool:
        if self._saved_snapshot is None:
            return True
        return self._collect_config().to_dict() != self._saved_snapshot

    def _confirm_discard_if_dirty(self) -> bool:
        if not self._is_dirty():
            return True
        result = QMessageBox.question(
            self,
            "Unsaved changes",
            "You have unsaved changes. Discard them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def _on_browse_folder(self) -> None:
        if self._is_dirty() and not self._confirm_discard_if_dirty():
            return
        start = str(self._config_dir or Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "Select config folder", start)
        if not chosen:
            return
        self._config_dir = Path(chosen)
        self._refresh_profile_ui()
        profiles = list_profiles(self._config_dir)
        if profiles:
            self._load_profile(profiles[0])
        else:
            self.profile_name_edit.setText("default")
            self._seed_form()
            self._mark_clean()

    def _on_new_folder(self) -> None:
        if self._is_dirty() and not self._confirm_discard_if_dirty():
            return
        start = str(self._config_dir or Path.cwd())
        parent = QFileDialog.getExistingDirectory(
            self, "Parent directory for new config folder", start
        )
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "New config folder", "Folder name:")
        if not ok or not name.strip():
            return
        new_dir = Path(parent) / name.strip()
        try:
            new_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            QMessageBox.warning(self, "Folder exists", f"Already exists:\n{new_dir}")
            return
        except OSError as exc:
            QMessageBox.warning(self, "Could not create folder", str(exc))
            return
        self._config_dir = new_dir
        self._refresh_profile_ui()
        self.profile_name_edit.setText("default")
        self._seed_form()
        self._mark_clean()

    def _load_profile(self, name: str) -> bool:
        """Load a profile by name from the current config folder."""
        if self._config_dir is None:
            return False
        stem = name[:-5] if name.lower().endswith(".json") else name.strip()
        path = profile_path(self._config_dir, stem)
        if not path.is_file():
            return False
        try:
            config = AudioEngineConfig.from_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not load profile", str(exc))
            return False
        self._set_profile_name(stem)
        self._apply_config(config)
        # Snapshot what the form actually holds (may coerce some fields).
        self._mark_clean()
        self.loaded_profile_label.setText(f"Loaded: {stem}.json")
        return True

    def _on_save(self, *, quiet: bool = False) -> bool:
        if self._config_dir is None:
            QMessageBox.information(
                self, "No config folder", "Select or create a config folder first."
            )
            return False
        name = self._current_profile_name()
        if not name:
            QMessageBox.warning(self, "Missing profile name", "Enter a profile name.")
            return False
        path = profile_path(self._config_dir, name)
        config = self._collect_config()
        if path.is_file():
            try:
                existing = AudioEngineConfig.from_file(path).to_dict()
            except Exception:
                existing = None
            if existing is not None and existing != config.to_dict():
                result = QMessageBox.question(
                    self,
                    "Overwrite profile?",
                    f"Overwrite existing profile “{name}”?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if result != QMessageBox.StandardButton.Yes:
                    return False
        try:
            config.to_file(path)
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self._refresh_profile_ui(selected=name)
        self._set_profile_name(name)
        self._mark_clean(config.to_dict())
        if not quiet:
            QMessageBox.information(self, "Saved", f"Saved profile to\n{path}")
        return True

    def _on_ok(self) -> None:
        if self._config_dir is not None and self._is_dirty():
            box = QMessageBox(self)
            box.setWindowTitle("Unsaved changes")
            box.setText("Save changes before closing?")
            box.setIcon(QMessageBox.Icon.Question)
            save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = box.addButton(
                "Discard", QMessageBox.ButtonRole.DestructiveRole
            )
            cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_btn:
                return
            if clicked is save_btn:
                if not self._on_save(quiet=True):
                    return
            elif clicked is not discard_btn:
                return
        self.accept()

    # --- device form ------------------------------------------------------

    def _apply_config(self, config: AudioEngineConfig) -> None:
        self._loading = True
        try:
            api_index = 0
            for i in range(self.api_combo.count()):
                data = self.api_combo.itemData(i)
                if data is not None and sd.query_hostapis(data)["name"] == config.api:
                    api_index = i
                    break
            if self.api_combo.currentIndex() == api_index:
                self._on_api_changed(api_index)
            else:
                self.api_combo.setCurrentIndex(api_index)

            # Exclusive is WASAPI-only; _on_api_changed clears it for other APIs.
            self.exclusive_check.setChecked(
                bool(config.exclusive) and "WASAPI" in config.api.upper()
            )

            # Prefer device name (stable across replug); fall back to index.
            device_id = config.stream.device
            device_name = config.device_name
            found = False
            for i in range(self.device_combo.count()):
                data = self.device_combo.itemData(i)
                if data and device_name and self.device_combo.itemText(i) == device_name:
                    self.device_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                for i in range(self.device_combo.count()):
                    data = self.device_combo.itemData(i)
                    if data and data[0] == device_id:
                        self.device_combo.setCurrentIndex(i)
                        found = True
                        break
            if not found and self.device_combo.count() > 0:
                label = (
                    f"“{device_name}” (index {device_id})"
                    if device_name
                    else f"index {device_id}"
                )
                QMessageBox.warning(
                    self,
                    "Device not found",
                    f"Saved device {label} is not available "
                    f"on this system for API “{config.api}”. Using the first device.",
                )
                self.device_combo.setCurrentIndex(0)
                self._on_device_changed(0)

            btn = self.channels_group.button(int(config.stream.channels))
            if btn is not None:
                btn.setChecked(True)

            text = str(int(config.stream.samplerate))
            idx = self.fs_combo.findText(text)
            if idx < 0:
                self.fs_combo.addItem(text)
                idx = self.fs_combo.findText(text)
            self.fs_combo.setCurrentIndex(idx)

            bs = config.stream.blocksize
            for i in range(self.blocksize_combo.count()):
                if self.blocksize_combo.itemData(i) == bs:
                    self.blocksize_combo.setCurrentIndex(i)
                    break

            lat = config.stream.latency
            for i in range(self.latency_combo.count()):
                if self.latency_combo.itemData(i) == lat:
                    self.latency_combo.setCurrentIndex(i)
                    break

            idx = self.dtype_combo.findData(config.stream.dtype)
            if idx >= 0:
                self.dtype_combo.setCurrentIndex(idx)

            self.clip_off_check.setChecked(config.stream.clip_off)
            self.dither_off_check.setChecked(config.stream.dither_off)
            self.never_drop_input_check.setChecked(config.stream.never_drop_input)
            self.prime_check.setChecked(
                config.stream.prime_output_buffers_using_stream_callback
            )

            has_advanced = any(
                [
                    bs != 0,
                    lat != "high",
                    config.stream.dtype != "float32",
                    config.stream.clip_off,
                    config.stream.dither_off,
                    config.stream.never_drop_input,
                    config.stream.prime_output_buffers_using_stream_callback,
                ]
            )
            self.advanced_group.setChecked(bool(has_advanced))
        finally:
            self._loading = False
            self._reset_test_button()

    def _on_api_changed(self, _index: int) -> None:
        api_id = self.api_combo.currentData()
        if api_id is None:
            return

        for btn in self.channels_group.buttons():
            self.channels_group.removeButton(btn)
        _clear_layout(self.channels_layout)

        api_name = str(sd.query_hostapis(api_id)["name"])
        self.exclusive_check.setEnabled("WASAPI" in api_name.upper())
        if "WASAPI" not in api_name.upper():
            self.exclusive_check.setChecked(False)

        self._channel_values = list_channels(api_id)
        self.channels_layout.addStretch()
        for ch in self._channel_values:
            rb = QRadioButton(str(ch))
            self.channels_group.addButton(rb, ch)
            rb.toggled.connect(self._reset_test_button)
            self.channels_layout.addWidget(rb)
        self.channels_layout.addStretch()

        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for dev_idx, dev_name, max_ch in list_output_devices(api_id):
            self.device_combo.addItem(dev_name, userData=(dev_idx, max_ch))
        self.device_combo.blockSignals(False)
        if self.device_combo.count() > 0:
            self.device_combo.setCurrentIndex(0)
            self._on_device_changed(0)

        has_devices = self.device_combo.count() > 0
        self.ok_button.setEnabled(has_devices)
        self.test_button.setEnabled(has_devices)
        self._reset_test_button()

    def _on_device_changed(self, _index: int) -> None:
        data = self.device_combo.currentData()
        if data is None:
            return
        dev_idx, max_ch = data
        btn = self.channels_group.button(max_ch)
        if btn is not None:
            btn.setChecked(True)
        # Prefer the device's native rate when the user picks a device.
        if not self._loading:
            rate = str(int(sd.query_devices(dev_idx)["default_samplerate"]))
            idx = self.fs_combo.findText(rate)
            if idx < 0:
                self.fs_combo.addItem(rate)
                idx = self.fs_combo.findText(rate)
            self.fs_combo.setCurrentIndex(idx)
        self._reset_test_button()

    def _connect_settings_reset(self) -> None:
        self.api_combo.currentIndexChanged.connect(self._reset_test_button)
        self.device_combo.currentIndexChanged.connect(self._reset_test_button)
        self.fs_combo.currentIndexChanged.connect(self._reset_test_button)
        self.exclusive_check.toggled.connect(self._reset_test_button)
        self.blocksize_combo.currentIndexChanged.connect(self._reset_test_button)
        self.latency_combo.currentIndexChanged.connect(self._reset_test_button)
        self.dtype_combo.currentIndexChanged.connect(self._reset_test_button)
        self.clip_off_check.toggled.connect(self._reset_test_button)
        self.dither_off_check.toggled.connect(self._reset_test_button)
        self.never_drop_input_check.toggled.connect(self._reset_test_button)
        self.prime_check.toggled.connect(self._reset_test_button)

    def _reset_test_button(self, *_args: object) -> None:
        if self._loading:
            return
        self.test_button.setText("Test")
        self.test_button.setStyleSheet("")
        self.test_button.setToolTip("")

    def _on_test(self) -> None:
        try:
            test_silent_output(self._collect_config())
        except Exception as exc:
            self.test_button.setText("Test failed")
            self.test_button.setStyleSheet(
                "background-color: #c62828; color: white; font-weight: bold;"
            )
            self.test_button.setToolTip(str(exc))
            return
        self.test_button.setText("Test OK")
        self.test_button.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold;"
        )
        self.test_button.setToolTip("Output stream opened successfully.")

    def _collect_config(self) -> AudioEngineConfig:
        api_id = self.api_combo.currentData()
        if api_id is None:
            raise RuntimeError("No host API selected")
        api_name = str(sd.query_hostapis(api_id)["name"])

        dev_data = self.device_combo.currentData()
        if dev_data is None:
            raise RuntimeError("No output device selected")
        device_id, max_ch = int(dev_data[0]), int(dev_data[1])

        channels = self.channels_group.checkedId()
        if channels <= 0:
            channels = max_ch

        return AudioEngineConfig(
            stream=StreamConfig(
                channels=int(channels),
                samplerate=float(self.fs_combo.currentText()),
                blocksize=int(self.blocksize_combo.currentData()),
                device=device_id,
                latency=cast(Latency, self.latency_combo.currentData()),
                dtype=cast(Dtype, self.dtype_combo.currentData()),
                clip_off=self.clip_off_check.isChecked(),
                dither_off=self.dither_off_check.isChecked(),
                never_drop_input=self.never_drop_input_check.isChecked(),
                prime_output_buffers_using_stream_callback=self.prime_check.isChecked(),
            ),
            api=api_name,
            device_name=self.device_combo.currentText(),
            exclusive=self.exclusive_check.isChecked()
            and "WASAPI" in api_name.upper(),
        )

    @staticmethod
    def select(
        config_dir: str | Path | None = None,
        profile: str | None = None,
        parent: QWidget | None = None,
    ) -> AudioEngineConfig:
        """Show the dialog modally and return the config.

        Exits the process with status 0 if the user cancels.
        """
        created_app = QApplication.instance() is None
        app = QApplication([]) if created_app else QApplication.instance()
        assert app is not None
        dialog = DeviceConfigSelector(
            parent, config_dir=config_dir, profile=profile
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        config = dialog._collect_config() if accepted else None
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        # Drop the QApplication we created so it cannot keep a Windows audio
        # session open that shadows the following PortAudio stream.
        if created_app:
            app.quit()
            app.processEvents()
        if config is None:
            sys.exit(0)
        return config
