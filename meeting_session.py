import asyncio
import glob
import os
import wave
import numpy as np
import whisper
import torch
from scipy.signal import resample
from collections import defaultdict
import livekit.rtc as rtc
from model_managers.mom_manager import MoMManager
from model_managers.transcription_manager import TranscriptionManager

import shutil


class MeetingSession:
    def __init__(self, room_name,mom_manager: MoMManager, transcription_manager: TranscriptionManager):
        self.room_name = room_name
        self.room = None
        self.mom_manager = mom_manager
        self.transcription_manager = transcription_manager

        self.audio_buffers = []
        self.participant_audio_map = defaultdict(list)
        self.participant_identity_map = {}

    def session_dir(self):
        return os.path.join("sessions", self.room_name)

    def session_file(self, filename):
        return os.path.join(self.session_dir(), filename)

    async def receive_audio(self, stream, sid):
        async for event in stream:
            frame = event.frame
            pcm = np.frombuffer(frame.data, dtype=np.int16)
            if np.max(np.abs(pcm)) > 0:
                self.participant_audio_map[sid].append(pcm)
                self.audio_buffers.append(pcm)
        await stream.aclose()

    def _on_track_subscribed(self, track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            stream = rtc.AudioStream(track)
            self.participant_identity_map[participant.sid] = participant.identity
            asyncio.create_task(self.receive_audio(stream, participant.sid))

    async def start(self, url, token):
        if os.path.exists(self.session_dir()):
         shutil.rmtree(self.session_dir())
        print(f"🗑️ Cleared previous session data for room: {self.room_name}")

        os.makedirs(self.session_dir(), exist_ok=True)
        self.room = rtc.Room()
        self.room.on("track_subscribed", self._on_track_subscribed)
        await self.room.connect(url, token)
        print(f"[{self.room_name}] Connected.")

    async def stop(self):
        if not self.room:
            return None
        await self.room.disconnect()
        self.room = None

        if not self.audio_buffers:
            return None

        await self.save_audio()
        return await self.process_audio()

    async def save_audio(self):
        data = np.concatenate(self.audio_buffers)
        with wave.open(self.session_file("audio.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(data.tobytes())

    def save_individual_speakers(self):
        speakers_dir = os.path.join(self.session_dir(), "speakers")
        os.makedirs(speakers_dir, exist_ok=True)
        wav_files = []
        for sid, pcms in self.participant_audio_map.items():
            identity = self.participant_identity_map.get(sid, f"user-{sid[:4]}")
            file = os.path.join(speakers_dir, f"{identity}.wav")
            with wave.open(file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(np.concatenate(pcms).tobytes())
            wav_files.append((identity, file))
        return wav_files

    def transcribe(self, wav_files):
        return self.transcription_manager.transcribe(wav_files, output_dir=self.session_dir())

    def generate_mom(self, transcript_path):
        return self.mom_manager.generate_from_transcript(transcript_path)

    async def process_audio(self):
        wavs = self.save_individual_speakers()
        transcript = self.transcribe(wavs)
        if not transcript.strip():
            return None
        mom = self.generate_mom(self.session_file("transcript.txt"))
        if mom:
            with open(self.session_file("mom.txt"), "w") as f:
                f.write(mom)
        return mom
