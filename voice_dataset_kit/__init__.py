"""Voice Dataset Kit - Convert long recordings into clean, segmented utterances.

This module patches torchaudio to fix deprecation warnings before importing speechbrain.
"""

import torchaudio

# Define what's exported when using 'from voice_dataset_kit import ...'
__all__ = ["apply_torchaudio_patch"]

def _patched_list_audio_backends():
    """Return list of available backends.

    This is a patched version that avoids the deprecated API.
    In modern torchaudio (2.1+), backends are handled automatically.
    """
    # In modern torchaudio, the dispatcher handles backends automatically
    # We return a mock list to satisfy speechbrain's check
    # The actual backend selection is handled by torchaudio internally
    return ["ffmpeg", "sox", "soundfile"]

def apply_torchaudio_patch():
    """Apply the monkey-patch to fix torchaudio deprecation warning.

    This patches torchaudio.list_audio_backends to avoid the deprecation warning
    that occurs when speechbrain internally calls this deprecated function.
    The function is deprecated and will be removed in torchaudio 2.9.
    """
    torchaudio.list_audio_backends = _patched_list_audio_backends

# Apply the patch immediately when this module is imported
apply_torchaudio_patch()