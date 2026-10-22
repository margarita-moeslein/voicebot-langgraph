import os

from dotenv import load_dotenv

load_dotenv()

LANGUAGE = os.getenv("VOICE_LANGUAGE", "de")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "YUdpWWny7k5yb4QCeweX")
TIMEZONE = "Europe/Berlin"

SAMPLE_RATE = 16000
THRESHOLD = 500
SILENCE_DURATION = 1.5
CHUNK_SIZE = 1024
