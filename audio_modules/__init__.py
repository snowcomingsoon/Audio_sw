"""
StudioEQ Master Pro — Audio DSP Modules Package.

Provides lazy/safe exports so importing individual DSP modules or top-level symbols
always succeeds cleanly without eager cross-module import side effects.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "detect_musical_key": ("vocal_autotune", "detect_musical_key"),
    "apply_vocal_autotune": ("vocal_autotune", "apply_vocal_autotune"),
    "detect_bpm": ("time_stretch_bpm", "detect_bpm"),
    "stretch_stems_to_bpm": ("time_stretch_bpm", "stretch_stems_to_bpm"),
    "enhance_and_reinforce_drums": ("drum_replacer", "enhance_and_reinforce_drums"),
    "enhance_and_resynthesize_bass": ("bass_enhancer", "enhance_and_resynthesize_bass"),
}

_SUBMODULES: set[str] = {
    "vocal_autotune",
    "time_stretch_bpm",
    "drum_samples_synth",
    "drum_replacer",
    "bass_enhancer",
}

__all__ = [*_LAZY_EXPORTS.keys(), *_SUBMODULES]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(f".{mod_name}", __name__)
        return getattr(module, attr_name)
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
