"""Shared catalog of supported TTS voices by provider."""

from __future__ import annotations

AVAILABLE_VOICES = {
    'openai': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
    'aws': ['Joanna', 'Matthew', 'Amy', 'Brian', 'Emma', 'Russell', 'Nicole', 'Raveena', 'Ivy', 'Kendra', 'Kimberly', 'Salli', 'Joey', 'Justin', 'Kevin'],
    'polly': ['Joanna', 'Matthew', 'Amy', 'Brian', 'Emma', 'Russell', 'Nicole', 'Raveena', 'Ivy', 'Kendra', 'Kimberly', 'Salli', 'Joey', 'Justin', 'Kevin'],
    'azure': ['en-US-AriaNeural', 'en-US-GuyNeural', 'en-US-JennyNeural', 'en-US-AmberNeural', 'en-US-AshleyNeural', 'en-US-BrandonNeural', 'en-US-ChristopherNeural', 'en-US-CoraNeural', 'en-US-DavisNeural', 'en-US-ElizabethNeural', 'en-US-EricNeural', 'en-US-JacobNeural', 'en-US-JaneNeural', 'en-US-JasonNeural', 'en-US-MichelleNeural', 'en-US-MonicaNeural', 'en-US-NancyNeural', 'en-US-RogerNeural', 'en-US-SaraNeural', 'en-US-SteffanNeural', 'en-US-TonyNeural'],
    'gcloud': ['en-US-Standard-A', 'en-US-Standard-B', 'en-US-Standard-C', 'en-US-Standard-D', 'en-US-Standard-E', 'en-US-Standard-F', 'en-US-Standard-G', 'en-US-Standard-H', 'en-US-Standard-I', 'en-US-Standard-J', 'en-US-Journey-D', 'en-US-Journey-F', 'en-US-News-K', 'en-US-News-L', 'en-US-News-M', 'en-US-News-N', 'en-US-Polyglot-1', 'en-US-Studio-M', 'en-US-Studio-O', 'en-US-Wavenet-A', 'en-US-Wavenet-B', 'en-US-Wavenet-C', 'en-US-Wavenet-D', 'en-US-Wavenet-E', 'en-US-Wavenet-F'],
    'elevenlabs': [
        ('21m00Tcm4TlvDq8ikWAM', 'Rachel'),
        ('AZnzlk1XvdvUeBnXmlld', 'Domi'),
        ('EXAVITQu4vr4xnSDxMaL', 'Bella'),
        ('ErXwobaYiN019PkySvjV', 'Antoni'),
        ('MF3mGyEYCl7XYWbV9V6O', 'Elli'),
        ('TxGEqnHWrfWFTfGW9XjX', 'Josh'),
        ('VR6AewLTigWG4xSOukaG', 'Arnold'),
        ('pNInz6obpgDQGcFmaJgB', 'Adam'),
        ('yoZ06aMxZJJ28mfd3POQ', 'Sam'),
    ],
    'deepgram': ['aura-asteria-en', 'aura-luna-en', 'aura-stella-en', 'aura-athena-en', 'aura-hera-en', 'aura-orion-en', 'aura-arcas-en', 'aura-perseus-en', 'aura-angus-en', 'aura-orpheus-en', 'aura-helios-en', 'aura-zeus-en'],
    'kittentts': ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'],
    'kokoro': [
        'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
        'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa',
        'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
        'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo',
        'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang',
        'ef_dora', 'em_alex', 'em_santa',
        'ff_siwis',
        'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi',
        'if_sara', 'im_nicola',
        'pf_dora', 'pm_alex', 'pm_santa',
    ],
    'system': [],
}

__all__ = ["AVAILABLE_VOICES"]
