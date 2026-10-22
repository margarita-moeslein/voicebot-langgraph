# voicebot-langgraph

Sprachassistent von Grund auf: **Whisper** (Speech-to-Text) → **LangGraph**-Agent mit Memory und Tool-Calling → **ElevenLabs** (Text-to-Speech).

```
START → audio_input → agent → audio_output → audio_input → ...
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # API-Keys eintragen
python main.py
```

## Konfiguration (`.env`)

| Variable | Bedeutung |
|---|---|
| `OPENAI_API_KEY` | Whisper + `gpt-4o-mini` |
| `ELEVEN_API_KEY` | ElevenLabs TTS |
| `VOICE_LANGUAGE` | Sprache für STT/TTS, Standard `de` |
| `ELEVEN_VOICE_ID` | ElevenLabs-Stimme |

## Aufbau

- `voicebot/agent.py` – LangGraph-Chatagent mit `get_date_and_time`-Tool (`ToolNode` + `tools_condition`)
- `voicebot/audio_io.py` – Mikrofonaufnahme mit Stille-Erkennung, Whisper, ElevenLabs
- `main.py` – Eltern-Graph mit `MemorySaver`

Das Projekt ist angelehnt an einen Artikel über den Nachbau des ElevenLabs-AI-Agents; der Graph wurde um fehlenden START-Edge und Tool-Node ergänzt.
