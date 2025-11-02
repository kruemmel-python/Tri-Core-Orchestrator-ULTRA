#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gemini Material Assessor – neue API (google-genai)
--------------------------------------------------
• Sicher: API-Key nur temporär im Speicher
• Unterstützt Streaming-Ausgabe
• Anzeige des generierten Textes im Streamlit-Frontend
• Keine Speicherung auf der Festplatte

Python 3.12
"""

import os
from google import genai
from google.genai import types
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Gemini Material Assessor (new API)", layout="wide", page_icon="🧪")
st.title("🧠 Gemini Material Assessor (google-genai API)")
st.caption("Bewertet Materialdaten aus Tri-Core-Orchestrator-ULTRA mit der neuen Gemini-API.")

# ---------------------------------------------------------------------
# API-Key-Eingabe
# ---------------------------------------------------------------------
api_key = st.sidebar.text_input(
    "🔑 Gemini API-Key",
    type="password",
    placeholder="AIzaSy…",
    help="Wird nur im Speicher gehalten und NICHT auf der Festplatte gespeichert."
)

if not api_key:
    st.warning("Bitte zuerst den Gemini-API-Key eingeben.")
    st.stop()

os.environ["GEMINI_API_KEY"] = api_key
client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------------
# Eingabe / Prompt
# ---------------------------------------------------------------------
st.sidebar.header("📄 Eingabe")
prompt_file = st.sidebar.text_input("Pfad zu gemini_prompt.md", value="gemini_prompt.md")

prompt_text = ""
if Path(prompt_file).exists():
    prompt_text = Path(prompt_file).read_text(encoding="utf-8")

prompt_edit = st.text_area("Prompt (editierbar):", value=prompt_text, height=300)

model_name = st.selectbox("Modell", ["gemini-2.5-flash", "gemini-2.5-pro-latest"], index=0)
image_size = st.selectbox("Bildgröße (falls generiert wird)", ["1K", "2K"], index=0)
use_search = st.checkbox("Google Search Tool aktivieren", value=True)
think_budget = st.number_input("Thinking Budget", min_value=0, max_value=1000, value=0, step=10)

# ---------------------------------------------------------------------
# Anfrage senden
# ---------------------------------------------------------------------
if st.button("🚀 Analyse starten"):
    with st.spinner("Sende Anfrage an Gemini (Streaming)…"):
        try:
            tools = []
            if use_search:
                tools.append(types.Tool(googleSearch=types.GoogleSearch()))

            gen_cfg = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=think_budget),
                image_config=types.ImageConfig(), 
                tools=tools,
            )


            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt_edit)],
                ),
            ]

            output_text = ""
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=gen_cfg,
            ):
                text = getattr(chunk, "text", "")
                if text:
                    output_text += text
                    st.write(text)  # Live-Streaming ins UI

            st.success("Antwort vollständig empfangen ✅")
            st.text_area("Gesamtausgabe:", value=output_text, height=400)

        except Exception as e:
            st.error(f"Fehler bei der Anfrage: {e}")
