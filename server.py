"""
VaaniCall — Real-time Speech-to-Speech Translation Server
Uses SeamlessM4T v2 for translation. Serves the web app on the same port.

Colab startup:
    !pip install fastapi uvicorn pyngrok pydub soundfile
    !python server.py &
    from pyngrok import ngrok
    print(ngrok.connect(7860))
"""
import asyncio, io, json, logging, os, random, string
from typing import Dict

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vaani")

app = FastAPI(title="VaaniCall")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load SeamlessM4T v2 ──────────────────────────────────────────────────────
log.info("Loading SeamlessM4T v2-large…")
from transformers import AutoProcessor, SeamlessM4Tv2Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

_proc  = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
_model = SeamlessM4Tv2Model.from_pretrained(
    "facebook/seamless-m4t-v2-large", torch_dtype=DTYPE
).to(DEVICE).eval()

OUTPUT_SR = 16000  # run_baseline_b.py expects/saves at 16000

LANG_CODES = {
    "English": "eng", "Hindi": "hin", "Telugu": "tel",
    "Marathi": "mar", "Tamil":  "tam", "Kannada": "kan",
    "Bengali": "ben", "Gujarati": "guj", "Punjabi": "pan",
}

log.info(f"✅  SeamlessM4T ready on {DEVICE}")

# ── Room state ───────────────────────────────────────────────────────────────
rooms: Dict[str, dict] = {}   # code → { users: { uid → { ws, lang } } }

def make_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

# ── Audio helpers ─────────────────────────────────────────────────────────────
def decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    """Accept webm / wav / ogg / any pydub-supported format → float32 mono 16 kHz."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(raw))
    seg = seg.set_channels(1).set_frame_rate(16_000)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32_768.0
    return samples, 16_000

def run_s2s(audio: np.ndarray, sr: int, src: str, tgt: str) -> bytes:
    # Match run_baseline_b.py exactly
    inputs = _proc(
        audios=audio, sampling_rate=sr,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model.generate(
            **inputs, 
            tgt_lang=LANG_CODES[tgt],
            return_intermediate_token_ids=False
        )
    wav = out[0].cpu().float().numpy().squeeze()
    
    buf = io.BytesIO()
    sf.write(buf, wav, OUTPUT_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()

# ── REST endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/room/create")
def create_room():
    code = make_code()
    rooms[code] = {"users": {}}
    log.info(f"Room created: {code}")
    return {"room_code": code}

# ── WebSocket ────────────────────────────────────────────────────────────────
async def broadcast(room: dict, payload: dict, exclude: str = None):
    for uid, u in list(room["users"].items()):
        if uid == exclude:
            continue
        try:
            await u["ws"].send_json(payload)
        except Exception:
            pass

@app.websocket("/ws/{code}/{uid}")
async def ws_handler(ws: WebSocket, code: str, uid: str):
    await ws.accept()

    if code not in rooms:
        rooms[code] = {"users": {}}
    room = rooms[code]
    room["users"][uid] = {"ws": ws, "lang": None}

    await broadcast(room, {"type": "room_update", "count": len(room["users"])})
    log.info(f"+ {uid} → {code}  ({len(room['users'])} users)")

    try:
        while True:
            msg = await ws.receive()

            # ── control message ──────────────────────────────────────────────
            if "text" in msg:
                d = json.loads(msg["text"])
                t = d.get("type")

                if t == "set_lang":
                    room["users"][uid]["lang"] = d["lang"]
                    await broadcast(room, {
                        "type": "lang_update", "user": uid, "lang": d["lang"]
                    })

                elif t == "ping":
                    await ws.send_json({"type": "pong"})

            # ── audio chunk ──────────────────────────────────────────────────
            elif "bytes" in msg:
                my_lang = room["users"][uid]["lang"]
                partners = [(pid, p) for pid, p in room["users"].items() if pid != uid]

                if not partners:
                    await ws.send_json({"type": "error", "msg": "No partner in room yet."})
                    continue

                pid, partner = partners[0]
                p_lang = partner["lang"]

                if not my_lang or not p_lang:
                    await ws.send_json({"type": "error",
                                        "msg": "Both users must select a language first."})
                    continue

                await ws.send_json({"type": "status", "msg": "translating"})

                try:
                    audio_arr, sr = await asyncio.to_thread(decode_audio, msg["bytes"])
                    wav_out       = await asyncio.to_thread(run_s2s, audio_arr, sr, my_lang, p_lang)

                    await partner["ws"].send_bytes(wav_out)
                    await partner["ws"].send_json({
                        "type": "incoming", "from_lang": my_lang, "to_lang": p_lang
                    })
                    await ws.send_json({"type": "status", "msg": "done"})

                except Exception as exc:
                    log.error(f"Translation error: {exc}")
                    await ws.send_json({"type": "error", "msg": str(exc)})

    except WebSocketDisconnect:
        room["users"].pop(uid, None)
        log.info(f"- {uid} ← {code}  ({len(room['users'])} users)")
        await broadcast(room, {"type": "room_update", "count": len(room["users"])})

# ── Serve the web app ────────────────────────────────────────────────────────
_web = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(_web):
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
