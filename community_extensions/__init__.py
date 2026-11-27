"""Community extensions for lab-ears (Layer L2)."""

from __future__ import annotations

from community_extensions.cross_modal_interlock import (
    SPEECH_REJECTION_MODES,
    CrossModalInterlockDetector,
    InterlockDecision,
    PipettingCommandWindow,
    SpeechUnfilteredRMSDetector,
    extract_voice_stripped_band_energies,
    run_benchmark,
    synthesize_broadband_speech_distractor,
)

__all__ = [
    "SPEECH_REJECTION_MODES",
    "CrossModalInterlockDetector",
    "InterlockDecision",
    "PipettingCommandWindow",
    "SpeechUnfilteredRMSDetector",
    "extract_voice_stripped_band_energies",
    "run_benchmark",
    "synthesize_broadband_speech_distractor",
]
