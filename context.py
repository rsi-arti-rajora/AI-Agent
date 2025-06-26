# context.py

from model_managers.mom_manager import MoMManager
from model_managers.transcription_manager import TranscriptionManager

def get_mom_manager() -> MoMManager:
    return MoMManager()

def get_transcription_manager() -> TranscriptionManager:
    return TranscriptionManager()
