#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini_assess.py
================
Sendet dein bereits erzeugtes Material-Paket (gemini_prompt.md + gemini_context.json)
unmittelbar an Gemini und streamt eine fundierte Einschätzung (Text + strukturiertes JSON).

Warum dieses Script?
- Nutzt NUR Textteile (keine ImageConfig etc.), vermeidet damit Pydantic-Fehler.
- Saubere Trennung von Prompt (Aufgabenstellung) und Kontext (Zahlen/Top-K u. Schema).
- Robustes Streaming inkl. Fallback, falls der Stream leer bleibt oder abbricht.
- Präzise, leise Fehlerbilder mit klaren Hinweisen (Rate Limits, Safety, etc.).

Python: 3.12  |  Abhängigkeit:  google-genai  (pip install google-genai)
Umgebung: Setze GEMINI_API_KEY vor dem Start (wird NICHT gespeichert).

Beispiel:
    set GEMINI_API_KEY=XXXX
    python gemini_assess.py --prompt gemini_prompt.md --context gemini_context.json --model gemini-1.5-pro-latest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Literal

from google import genai
from google.genai import types


# ----------------------------- I/O-Helfer -----------------------------

def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    return p.read_text(encoding=encoding)

def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------- Konfiguration -----------------------------

@dataclass
class RunConfig:
    model: str = "gemini-1.5-pro-latest"
    prompt_path: str = "gemini_prompt.md"
    context_path: str = "gemini_context.json"
    temperature: float = 0.2
    top_k: int = 32
    top_p: float = 0.95
    stream: bool = True
    max_retries: int = 2
    use_google_search: bool = False  # optionales Tool

    @staticmethod
    def from_args() -> "RunConfig":
        ap = argparse.ArgumentParser(description="Gemini Materialbewertung aus lokalem Prompt+Kontext.")
        ap.add_argument("--prompt", type=str, default="gemini_prompt.md", help="Pfad zur Prompt-Datei (Markdown)")
        ap.add_argument("--context", type=str, default="gemini_context.json", help="Pfad zur Kontext-JSON")
        ap.add_argument("--model", type=str, default="gemini-1.5-pro-latest", help="Gemini-Modell-ID")
        ap.add_argument("--temp", type=float, default=0.2, help="Sampling-Temperatur")
        ap.add_argument("--top-k", type=int, default=32, help="Top-K")
        ap.add_argument("--top-p", type=float, default=0.95, help="Top-P")
        ap.add_argument("--no-stream", action="store_true", help="Streaming deaktivieren (synchroner Aufruf)")
        ap.add_argument("--retries", type=int, default=2, help="Max. automatische Retries bei leeren/abgebrochenen Streams")
        ap.add_argument("--google-search", action="store_true", help="Google Search Tool aktivieren")
        args = ap.parse_args()
        return RunConfig(
            model=args.model,
            prompt_path=args.prompt,
            context_path=args.context,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            stream=(not args.no_stream),
            max_retries=max(0, args.retries),
            use_google_search=bool(args.google_search),
        )


# ----------------------------- Anfrageaufbau -----------------------------

def build_contents(prompt_text: str, context_json: dict) -> list[types.Content]:
    """
    Wir schicken:
    - Teil 1: "Rolle/Instruktion" (dein Prompt-Text)
    - Teil 2: "Datenblock" (Kontext als sauber formatiertes JSON)
    """
    context_compact = json.dumps(context_json, ensure_ascii=False, indent=2)

    return [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "Du bist Materialwissenschafts-Experte. "
                        "Lies die Aufgabenstellung und dann die Daten. "
                        "Gib zuerst eine Kurzbegründung, dann GENAU das geforderte JSON.\n\n"
                        "=== AUFGABENSTELLUNG ===\n"
                        f"{prompt_text}\n"
                    )
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "=== DATENKONTEXT (JSON) ===\n"
                        f"{context_compact}\n"
                    )
                ),
            ],
        ),
    ]


def build_config(cfg: RunConfig) -> types.GenerateContentConfig:
    tools = []
    if cfg.use_google_search:
        tools = [types.Tool(googleSearch=types.GoogleSearch())]

    # KEINE ImageConfig hier – wir senden nur Text.
    return types.GenerateContentConfig(
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        tools=tools or None,
        # Optional könnten wir hier safety_settings setzen; weglassen = Default
    )


# ----------------------------- Execution/Streaming -----------------------------

def stream_reply(client: genai.Client, cfg: RunConfig, contents: list[types.Content]) -> str:
    """
    Streamt die Antwort. Falls der Stream leer ist / keine Parts kommen (z. B. finish_reason==2),
    probiert die Funktion (bis zu max_retries) erneut, und fällt bei Bedarf auf Non-Streaming zurück.
    """
    aggregate = []
    attempt = 0

    while True:
        attempt += 1
        empty = True
        try:
            for chunk in client.models.generate_content_stream(
                model=cfg.model,
                contents=contents,
                config=build_config(cfg),
            ):
                # Manche Chunks haben kein .text; also defensiv sammeln:
                part = getattr(chunk, "text", None)
                if part:
                    empty = False
                    aggregate.append(part)
                    print(part, end="", flush=True)  # Live-Ausgabe
            print("")  # Zeilenumbruch am Ende
        except Exception as e:
            # Rate-Limits / Netzwerk / API-Fehler -> kurzer Backoff
            sys.stderr.write(f"\n[Warn] Streaming-Fehler: {e}\n")
            if attempt <= cfg.max_retries:
                time.sleep(1.0 * attempt)
                continue
            sys.stderr.write("[Info] Wechsle auf Non-Streaming-Fallback …\n")
            return nonstream_reply(client, cfg, contents)

        # Leerer Stream? -> Retry oder Fallback
        if empty:
            sys.stderr.write("\n[Warn] Leerer Stream (keine Parts).")
            if attempt <= cfg.max_retries:
                sys.stderr.write(f" Wiederhole (Versuch {attempt}/{cfg.max_retries}) …\n")
                time.sleep(0.6 * attempt)
                continue
            sys.stderr.write(" Fallback auf Non-Streaming.\n")
            return nonstream_reply(client, cfg, contents)

        return "".join(aggregate)


def nonstream_reply(client: genai.Client, cfg: RunConfig, contents: list[types.Content]) -> str:
    """
    Nicht-streamende Alternative. Gibt kompletten Text zurück oder wirft Exception.
    """
    resp = client.models.generate_content(
        model=cfg.model,
        contents=contents,
        config=build_config(cfg),
    )
    # Der SDK-Response kann mehrere Candidates haben; wir nehmen text-Feld, wenn vorhanden.
    text = getattr(resp, "text", None)
    if not text:
        # Falls kein text (z. B. wegen Safety-Block), versuchen wir aus parts/candidates zu lesen:
        try:
            cands = getattr(resp, "candidates", None) or []
            buf = []
            for c in cands:
                for p in getattr(c, "content", {}).get("parts", []) or []:
                    t = getattr(p, "text", None)
                    if t:
                        buf.append(t)
            if buf:
                return "".join(buf)
        except Exception:
            pass
        raise RuntimeError("Antwort enthält keinen Text (möglicherweise Safety-Filter oder leere Kandidaten).")
    print(text)
    return text


# ----------------------------- Main -----------------------------

def main() -> None:
    cfg = RunConfig.from_args()

    # API-Key (nur aus ENV)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.stderr.write("Fehler: Bitte setze die Umgebungsvariable GEMINI_API_KEY.\n")
        sys.exit(2)

    # Dateien laden
    prompt_text = read_text(cfg.prompt_path)
    context_json = read_json(cfg.context_path)

    # Client
    client = genai.Client(api_key=api_key)

    # Anfrage bauen
    contents = build_contents(prompt_text, context_json)

    # Stream oder Non-Stream
    if cfg.stream:
        _ = stream_reply(client, cfg, contents)
    else:
        _ = nonstream_reply(client, cfg, contents)


if __name__ == "__main__":
    main()
