"""Hugging Face / Gradio Web Demo for Jerry Voice AI Interviewer."""
from __future__ import annotations

import os
import gradio as gr
from interview_agent.questions import QUESTIONS, GREETING, SIGNOFF

def interview_preview(question_num: int):
    idx = max(0, min(question_num - 1, len(QUESTIONS) - 1))
    q = QUESTIONS[idx]
    return f"**Question {q['id']} of {len(QUESTIONS)}:** {q['text']}"

with gr.Blocks(title="Jerry - Voice AI Interview Agent") as demo:
    gr.Markdown("# 🎙️ Jerry: Full-Duplex Voice AI Interviewer")
    gr.Markdown("Conduct structured technical interviews with Groq Whisper STT, LangGraph, and Edge TTS.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📜 System Overview")
            gr.Markdown(f"**Greeting**: _{GREETING}_")
            gr.Markdown(f"**Signoff**: _{SIGNOFF}_")
            
            q_slider = gr.Slider(minimum=1, maximum=len(QUESTIONS), step=1, value=1, label="Select Question #")
            q_box = gr.Markdown(interview_preview(1))
            q_slider.change(fn=interview_preview, inputs=[q_slider], outputs=[q_box])
            
        with gr.Column():
            gr.Markdown("### ⚙️ Quickstart Command")
            gr.Code(
                value="git clone https://github.com/Vikashy05/-voice-interview-agent.git\ncd -voice-interview-agent\npip install -r requirements.txt\npython -m interview_agent.main",
                language="shell",
                label="Run Locally",
            )

if __name__ == "__main__":
    demo.launch()
