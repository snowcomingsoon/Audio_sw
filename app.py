#!/usr/bin/env python3
"""
Server Web locale per lo Studio EQ & Mastering Suite.
Offre:
- Streaming diretto della traccia Originale e della traccia Masterizzata per ascolto A/B istantaneo
- Motore Web Audio API + Analizzatore di Spettro FFT in tempo reale + Curva EQ parametrica interattiva
- Endpoint API (/api/render) per processare ed esportare nuove tracce a 64-bit tramite dsp_engine.py
"""

import os
import json
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from dsp_engine import process_track, DEFAULT_SALA_PROVE_EQ

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")

ORIGINAL_UPLOAD = r"C:/Users/Lorenzo/.gemini/antigravity/brain/008e9ef7-9010-4af0-b7ad-d6056844c4e0/.user_uploaded/media_1793370806307_ad5c61d4.mpeg"
ORIGINAL_COPY = os.path.join(OUTPUT_DIR, "traccia_originale_sala.mp3")

os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.exists(ORIGINAL_UPLOAD) and not os.path.exists(ORIGINAL_COPY):
    shutil.copyfile(ORIGINAL_UPLOAD, ORIGINAL_COPY)


class StudioHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.path = "/index.html"
        return super().do_GET()

    def translate_path(self, path):
        parsed = urlparse(path).path
        if parsed.startswith("/output/"):
            rel = parsed[len("/output/"):]
            return os.path.join(OUTPUT_DIR, rel)
        if parsed == "/" or parsed == "/index.html":
            return os.path.join(STATIC_DIR, "index.html")
        return os.path.join(STATIC_DIR, parsed.lstrip("/"))

    def do_POST(self):
        parsed = urlparse(self.path).path
        if parsed == "/api/render":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

            eq_bands = body.get("eq_bands", DEFAULT_SALA_PROVE_EQ)
            stereo_width = float(body.get("stereo_width", 1.22))
            exciter_drive = float(body.get("exciter_drive", 0.22))
            enable_dynamic_mud = bool(body.get("enable_dynamic_mud", True))

            out_wav = os.path.join(OUTPUT_DIR, "traccia_equalizzata_custom.wav")
            out_mp3 = os.path.join(OUTPUT_DIR, "traccia_equalizzata_custom.mp3")

            stats = process_track(
                input_path=ORIGINAL_COPY,
                output_wav=out_wav,
                output_mp3=out_mp3,
                eq_bands=eq_bands,
                enable_dynamic_mud=enable_dynamic_mud,
                enable_exciter=True,
                enable_mid_side=True,
                stereo_width=stereo_width,
                exciter_drive=exciter_drive,
            )

            resp = json.dumps({
                "status": "ok",
                "wav_url": "/output/traccia_equalizzata_custom.wav",
                "mp3_url": "/output/traccia_equalizzata_custom.mp3",
                "stats": stats,
            }).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        self.send_error(404, "Endpoint non trovato")


if __name__ == "__main__":
    port = 8890
    print(f"Studio EQ & Mastering Suite attiva su: http://localhost:{port}")
    server = HTTPServer(("127.0.0.1", port), StudioHandler)
    server.serve_forever()
