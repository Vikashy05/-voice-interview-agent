# Full-Duplex Voice Interview Agent

<p align="center">
  <img src="assets/banner.jpg" alt="Jerry Voice AI Interviewer Banner" width="100%" />
</p>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-FF6F61?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-STT%20%26%20LLM-f34b7d?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)


This project is a high-performance Python voice interview agent named **Jerry**. It conducts structured Machine Learning technical interviews using a full-duplex speech engine: listening continuously through the microphone, transcribing answers with Groq Whisper, generating instant streaming turns with Groq LLMs, and synthesizing natural speech with Microsoft Edge TTS.

The main goal is natural interview flow: Jerry can listen while speaking,
distinguish short backchannels from real interruptions, stop playback quickly
when the candidate cuts in, and save a written record of the session.

## What It Does

- Runs a seven-question ML fundamentals interview.
- Supports full-duplex voice interaction by capturing microphone audio while
  TTS playback is still running.
- Uses a two-level interruption system: a fast audio gate stops speech quickly,
  then transcription decides whether the user really wanted the floor.
- Streams LLM output into speech chunks so Jerry starts speaking before the full
  response is complete.
- Saves local interview artifacts under `interviews/` when a session runs.
- Optionally writes answers to PostgreSQL when `DATABASE_URL` is configured.
- Includes a small static web dashboard in `web/` for reviewing generated
  interview output.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `interview_agent/main.py` | CLI entry point for running an interview. |
| `interview_agent/duplex.py` | Full-duplex microphone capture, transcription, and playback coordination. |
| `interview_agent/streaming.py` | LLM token streaming, phrase chunking, TTS synthesis, and cancellation. |
| `interview_agent/interrupt.py` | Fast barge-in detection and interruption verdict handling. |
| `interview_agent/audio.py` | Microphone, playback, VAD, and low-level audio helpers. |
| `interview_agent/brain.py` | Groq client calls for STT, interview responses, and summaries. |
| `interview_agent/questions.py` | The fixed interview question script. |
| `interview_agent/recorder.py` | Local session output writer for answers, transcripts, metrics, and notes. |
| `interview_agent/db.py` | Optional PostgreSQL persistence, one table per interview run. |
| `interview_agent/config.py` | Runtime configuration and audio/model tuning constants. |
| `miccheck.py` | Microphone calibration utility. |
| `web/` | Static review dashboard assets. |

## Requirements

- Python 3.11 or newer.
- A working microphone and speaker output device.
- A Groq API key for both LLM and Whisper transcription.
- Optional: PostgreSQL if you want database persistence.

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Local Configuration

Secrets are intentionally not committed. The repository ignores `.env`,
`.env.*`, and `.env.example`, so create a local `.env` file only on your own
machine.

Required:

```env
GROQ_API_KEY=your_groq_api_key
```

Optional PostgreSQL storage:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/interview_agent
```

If `DATABASE_URL` is not set, the app still runs and writes local files under
`interviews/`.

## Running The App

Calibrate the microphone first:

```bash
python miccheck.py
```

Run the default full-duplex interview:

```bash
python -m interview_agent.main
```

Useful options:

```bash
python -m interview_agent.main --half-duplex
python -m interview_agent.main --no-notes
python -m interview_agent.main --debug-barge
python -m interview_agent.main --outdir interviews/manual-test
```

## Output

Each full-duplex run writes text-only artifacts:

```text
interviews/<timestamp>_<session_id>/
  answers.md
  results.json
  transcript.txt
  conversation.txt
  error.txt          # only when a run crashes
```

No raw audio is saved.

## Architecture

The default path is concurrent rather than turn-by-turn:

```mermaid
flowchart TD
    subgraph Input ["🎙️ Audio Capture"]
        Mic["Microphone Input"] --> VAD["Voice Activity Detector (VAD)"]
        VAD --> STT["Groq Whisper (STT)"]
    end

    subgraph Core ["🧠 Intelligence & Orchestration"]
        STT --> Graph["LangGraph Interview Engine"]
        Graph --> Brain["Groq LLM Engine"]
        BargeGate["Fast Interruption Gate"] -.->|"Interrupt Event"| AudioPlayer
    end

    subgraph Output ["🔊 Audio Synthesis & Output"]
        Brain --> Chunker["Streaming Chunker"]
        Chunker --> TTS["Edge TTS"]
        TTS --> AudioPlayer["Speaker Playback"]
    end

    Mic -.->|"Barge-in Monitoring"| BargeGate
```

While Jerry is speaking, microphone capture continues. If the candidate speaks
over playback, the interruption gate stops audio quickly and marks the current
generation stale so old TTS chunks cannot resume after the user takes the turn.

The interview state is managed through LangGraph. Local memory keeps the recent
conversation context for follow-up prompts, while `recorder.py` persists the
human-readable transcript and structured metrics.

## Database Behavior

Database storage is optional. When `DATABASE_URL` is present, every interview
run creates its own PostgreSQL table named from the run id, for example:

```text
interview_20260826_144759_a1b2c3
```

Each row stores:

- question number
- question text
- candidate answer

The fixed question list still lives in `interview_agent/questions.py`.

## Security Notes

- `.env` and `.env.example` are not tracked.
- Do not commit API keys, database passwords, transcripts from real candidates,
  or generated interview output.
- If a key was ever committed or shared in plain text, rotate it before pushing
  the repository publicly.
