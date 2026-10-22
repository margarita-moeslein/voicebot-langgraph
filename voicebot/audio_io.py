import io

import numpy as np
import sounddevice as sd
from elevenlabs import VoiceSettings, play
from elevenlabs.client import ElevenLabs
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from openai import OpenAI
from scipy.io.wavfile import write

from . import config

openai_client = OpenAI()
elevenlabs_client = ElevenLabs()  # reads ELEVEN_API_KEY from the environment


def _record_until_silence() -> np.ndarray:
    chunks = []
    silent = 0
    max_silent = config.SILENCE_DURATION * config.SAMPLE_RATE / config.CHUNK_SIZE

    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16") as stream:
        print("Warte auf Sprache...")
        while True:
            chunk, _ = stream.read(config.CHUNK_SIZE)
            if np.abs(chunk).max() > config.THRESHOLD:
                print("Stimme erkannt, Aufnahme läuft.")
                chunks.append(chunk)
                break

        while True:
            chunk, _ = stream.read(config.CHUNK_SIZE)
            chunks.append(chunk)
            silent = silent + 1 if np.abs(chunk).max() < config.THRESHOLD else 0
            if silent > max_silent:
                print("Stille erkannt, Aufnahme beendet.")
                break

    return np.concatenate(chunks, axis=0)


def record_audio_until_silence(state: MessagesState):
    """Record from the microphone and transcribe with Whisper."""
    audio = _record_until_silence()

    buffer = io.BytesIO()
    write(buffer, config.SAMPLE_RATE, audio)
    buffer.seek(0)
    buffer.name = "audio.wav"

    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=buffer,
        language=config.LANGUAGE,
    )
    print("Transkription:", transcription.text)
    return {"messages": [HumanMessage(content=transcription.text)]}


def play_audio(state: MessagesState):
    """Speak the last agent message with ElevenLabs."""
    text = state["messages"][-1].content.replace("**", "")

    audio = elevenlabs_client.text_to_speech.convert(
        voice_id=config.VOICE_ID,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        language_code=config.LANGUAGE,
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    play(audio)
    return {}
