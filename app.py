# app.py - AI Driver Monitoring System (main file)
import json
import math
import threading
import time

import cv2
import requests

import av
import streamlit.components.v1 as components
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from advanced import DriverMonitor, build_report_html, fmt_time, samples_csv
from detector import extract_hands, initialize_face_mesh, initialize_hands, process_frame
from driver_utils import draw_overlay

st.set_page_config(
    page_title="AI Driver Monitoring System",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 50% 50%, #0d1117 0%, #030712 100%); color: #f3f4f6; }
.main-title { font-size: 1.6rem; font-weight: 800; text-align: center; margin-bottom: 14px;
  background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
div[data-testid="stMetric"] { background: rgba(17,24,39,0.75); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px; padding: 10px 14px; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 800 !important; color: #38bdf8 !important; }
div[data-testid="stMetricLabel"] { color: #9ca3af !important; text-transform: uppercase; }
.status-card { padding: 14px; border-radius: 14px; text-align: center; font-size: 1.1rem;
  font-weight: 800; margin-bottom: 12px; }
.status-normal { background: rgba(6,78,59,0.6); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.status-warning { background: rgba(120,53,15,0.7); color: #fbbf24; border: 1px solid rgba(251,191,36,0.5); }
.status-danger { background: rgba(127,29,29,0.8); color: #f87171; border: 1px solid rgba(248,113,113,0.6); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.status-card { box-shadow: 0 0 22px rgba(56,189,248,0.10); letter-spacing: 1px; }
.status-danger { box-shadow: 0 0 28px rgba(248,113,113,0.45); }
.status-warning { box-shadow: 0 0 24px rgba(251,191,36,0.35); }
.hud { background: rgba(10,15,28,0.85); border: 1px solid rgba(56,189,248,0.25); border-radius: 18px;
  padding: 12px; margin: 8px 0; box-shadow: 0 0 22px rgba(56,189,248,0.10) inset, 0 0 16px rgba(56,189,248,0.12); }
.hud-row { display: flex; gap: 14px; align-items: center; justify-content: center; flex-wrap: wrap; }
.hud-bars { flex: 1; min-width: 150px; }
.lbl { font-size: 11px; color: #9ca3af; display: flex; justify-content: space-between; margin: 6px 0 3px; letter-spacing: 1px; }
.bar { height: 8px; border-radius: 6px; background: #1f2937; overflow: hidden; }
.bar > div { height: 100%; border-radius: 6px; }
.coach { padding: 10px 14px; border-radius: 14px; font-size: 0.95rem; margin: 6px 0; }
.coach-ok { background: rgba(8,47,73,0.6); border: 1px solid rgba(56,189,248,0.35); color: #7dd3fc; }
.coach-tip { background: rgba(120,53,15,0.55); border: 1px solid rgba(251,191,36,0.45); color: #fcd34d; }
.coach-break { background: rgba(127,29,29,0.7); border: 1px solid rgba(248,113,113,0.6); color: #fca5a5; }
</style>
""", unsafe_allow_html=True)



try:                                   # optional: phone location for owner alerts
    from streamlit_js_eval import get_geolocation
    HAS_GEO = True
except Exception:
    HAS_GEO = False

st.markdown("""
<style>
.block-container { padding: 2.2rem 1rem 1rem 1rem !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stHorizontalBlock"] { gap: 0.8rem; }
.status-card { padding: 10px 12px !important; font-size: 1.05rem !important; margin-bottom: 6px !important; }
.topbar { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
  padding: 8px 14px; margin-bottom: 8px; border-radius: 14px;
  background: linear-gradient(90deg, rgba(15,23,42,0.95), rgba(30,27,75,0.85));
  border: 1px solid rgba(129,140,248,0.25); box-shadow: 0 0 22px rgba(99,102,241,0.15); }
.brand { display: flex; align-items: center; gap: 10px; }
.logo { font-size: 26px; }
.bt { font-size: 16px; font-weight: 800; letter-spacing: 2px;
  background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.bs { font-size: 10px; color: #9ca3af; letter-spacing: 1px; }
.pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 700; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); color: #e5e7eb; }
.pill i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.tiles { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.tile { flex: 1 1 30%; min-width: 90px; background: rgba(15,23,42,0.85); border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px; padding: 8px 10px; }
.tl { font-size: 10px; letter-spacing: 1.5px; color: #9ca3af; }
.tv { font-size: 20px; font-weight: 800; }
button[data-baseweb="tab"] { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.cluster { background: linear-gradient(180deg, #05080f 0%, #0a0f1a 100%); border: 1px solid #1e293b; border-radius: 22px;
  padding: 10px 12px 8px; box-shadow: 0 0 0 1px #0b1220 inset, 0 10px 30px rgba(0,0,0,0.6); }
.cl-danger { animation: clpulse 1s ease-in-out infinite; }
@keyframes clpulse { 0%,100% { box-shadow: 0 0 0 1px #0b1220 inset, 0 0 14px rgba(239,68,68,0.25); }
                     50% { box-shadow: 0 0 0 1px #0b1220 inset, 0 0 36px rgba(239,68,68,0.75); } }
.cl-status { text-align: center; font-size: 15px; font-weight: 700; letter-spacing: 3px; margin: 0 0 6px; }
.tell { display: flex; justify-content: space-between; gap: 2px; margin: 6px 0 2px; }
.tl-i { flex: 1; text-align: center; font-size: 17px; line-height: 1; }
.tl-i small { display: block; font-size: 7.5px; letter-spacing: 1px; color: #64748b; margin-top: 3px; }
.tl-off { filter: grayscale(1) opacity(0.22); }
.trip { display: flex; border-top: 1px solid #1e293b; padding-top: 6px; margin-top: 6px; }
.trip div { flex: 1; text-align: center; }
.trip b { display: block; font-size: 14px; color: #e2e8f0; font-weight: 700; }
.trip span { font-size: 8px; letter-spacing: 1.5px; color: #64748b; }
</style>
""", unsafe_allow_html=True)

def _tg_config():
    """Read Telegram settings from Streamlit Secrets (never store them in GitHub)."""
    try:
        return st.secrets["TELEGRAM_BOT_TOKEN"], str(st.secrets["TELEGRAM_CHAT_ID"])
    except Exception:
        return None, None


def send_telegram(text, jpg=None):
    """Send a message (and optional photo) to the owner. Returns (ok, info)."""
    token, chat_id = _tg_config()
    if not token or not chat_id:
        return False, "Telegram is not set up yet (add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Secrets)."
    base = f"https://api.telegram.org/bot{token}"
    try:
        if jpg is not None:
            r = requests.post(f"{base}/sendPhoto", data={"chat_id": chat_id, "caption": text},
                              files={"photo": ("driver.jpg", jpg, "image/jpeg")}, timeout=10)
        else:
            r = requests.post(f"{base}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=10)
        data = r.json()
        if data.get("ok"):
            return True, "Sent"
        return False, data.get("description", "Telegram refused the message")
    except Exception as e:
        return False, f"Could not reach Telegram ({type(e).__name__})"


def send_telegram_location(lat, lon):
    """Send a map pin the owner can tap to open in Maps."""
    token, chat_id = _tg_config()
    if not token or not chat_id:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendLocation",
                      data={"chat_id": chat_id, "latitude": lat, "longitude": lon}, timeout=10)
    except Exception:
        pass


class DriverProcessor(VideoProcessorBase):
    """Keeps the video smooth: recv() only stores the newest frame and returns at once.
    A background worker thread does the heavy face / hand analysis (about 8 times a second)
    on the newest frame only, so slow analysis can never make the video lag."""

    ANALYSIS_HZ = 8

    def __init__(self):
        self.face_mesh = initialize_face_mesh()
        self.hands = initialize_hands()
        self.monitor = DriverMonitor()
        self.last_img = None
        self.fps = 0.0
        self.proc_ms = 0.0
        self._prev = time.time()
        self._hands_pts = []
        self._tick = 0
        self._last_proc = 0.0
        self._latest = None
        self._last_frame_t = time.time()
        self._lock = threading.Lock()
        self._worker = None
        self._start_worker()

    def _start_worker(self):
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self):
        period = 1.0 / self.ANALYSIS_HZ
        while time.time() - self._last_frame_t < 5.0:      # stops by itself when the camera stops
            t0 = time.time()
            with self._lock:
                img, self._latest = self._latest, None
            if img is None:
                time.sleep(0.02)
                continue
            try:
                self._analyze(img, t0)
            except Exception:
                pass
            self.proc_ms = 0.8 * self.proc_ms + 0.2 * (time.time() - t0) * 1000
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _analyze(self, img, now):
        dt = (now - self._last_proc) if self._last_proc else 1.0 / self.ANALYSIS_HZ
        self._last_proc = now
        step = min(dt * 30.0, 6.0)                       # "frames at 30 fps" that passed
        _, results = process_frame(img, self.face_mesh)
        lm = None
        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
        self._tick += 1
        if self._tick % 3 == 0:                          # hands are checked about 3 times per second
            self._hands_pts = extract_hands(img, self.hands)
        h, w = img.shape[:2]
        self.monitor.update(lm, now, step, self._hands_pts, w / h)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        self._last_frame_t = now
        inst = 1 / (now - self._prev) if now > self._prev else 0
        self.fps = 0.8 * self.fps + 0.2 * inst
        self._prev = now

        with self._lock:
            self._latest = img.copy()                    # newest frame for the worker (older ones are dropped)
        if self._worker is None or not self._worker.is_alive():
            self._start_worker()

        m = self.monitor
        img = draw_overlay(img, m.status, self.fps)
        if m.calib.active:
            cv2.putText(img, "CALIBRATING - look at the road", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self.last_img = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")


VOICE = {
    ("Danger", "Eyes closed"): "Danger! Your eyes are closed. Wake up!",
    ("Danger", "Yawning"): "Danger! You are very sleepy. Please take a break.",
    ("Danger", "Head down"): "Danger! Keep your head up and look at the road!",
    ("Danger", "Face not visible"): "Danger! I cannot see your face. Look at the road!",
    ("Warning", "Eyes closed"): "Warning. Keep your eyes open.",
    ("Warning", "Yawning"): "Warning. You seem tired.",
    ("Warning", "Head down"): "Warning. Please look up.",
    ("Warning", "Looking away"): "Warning. Eyes on the road.",
    ("Danger", "Looking away"): "Danger! Look at the road!",
    ("Warning", "Using phone"): "Warning. Please put your phone down.",
    ("Danger", "Using phone"): "Danger! Stop using your phone now!",
    ("Warning", "Phone call"): "Warning. Please end the call or use hands free.",
    ("Danger", "Phone call"): "Danger! Put the phone down and focus on driving!",
    ("Warning", "Eating or drinking"): "Warning. Please keep your hands on the wheel.",
    ("Danger", "Eating or drinking"): "Danger! Stop eating and keep both hands on the wheel!",
}

ALERT_JS = """<script>/*__NONCE__*/
// 1) beep
if (__BEEP__) { try {
  const C = window.AudioContext || window.webkitAudioContext;
  const ctx = new C(); ctx.resume();
  for (let i = 0; i < __BEEPS__; i++) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'square'; o.frequency.value = __FREQ__; g.gain.value = 0.5;
    o.connect(g); g.connect(ctx.destination);
    const t = ctx.currentTime + i * 0.35; o.start(t); o.stop(t + 0.25);
  }
} catch (e) {} }
// 2) full-screen flash
if (__FLASH__) { try {
  const d = window.parent.document;
  if (!d.getElementById('dms-style')) {
    const st = d.createElement('style'); st.id = 'dms-style';
    st.textContent = '@keyframes dmsflash{0%{opacity:0}12%{opacity:.8}30%{opacity:.15}50%{opacity:.75}100%{opacity:0}}';
    d.head.appendChild(st);
  }
  const old = d.getElementById('dms-flash'); if (old) old.remove();
  const el = d.createElement('div'); el.id = 'dms-flash';
  el.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none;opacity:0;background:__COLOR__;animation:dmsflash 1.3s ease-out forwards;';
  d.body.appendChild(el);
} catch (e) {} }
// 3) voice warning (uses the main page's speech engine so it survives iframe refresh)
if (__VOICE__) { try {
  const W = window.parent, S = W.speechSynthesis, now = Date.now();
  // speak if idle, or if the engine looks stuck (more than 5 s since last start)
  if (S && (!S.speaking || now - (W.__dmsVoiceAt || 0) > 5000)) {
    S.cancel(); S.resume();
    const u = new W.SpeechSynthesisUtterance(__TEXT__);
    u.rate = 1.0; u.volume = 1; u.lang = 'en-US';
    W.__dmsU = u; W.__dmsVoiceAt = now;      // keep a reference so it is not garbage-collected
    W.setTimeout(function () { S.speak(u); }, 120);
  }
} catch (e) {} }
// 4) vibration
if (__VIBRATE__) { try { navigator.vibrate(__VIB__); } catch (e) {} }
</script>"""


def fire_alert(box, status, reason, beep=True, voice=True, flash=True, vibrate=True):
    """Phone alert: beep + spoken warning + full-screen flash + vibration."""
    danger = status == "Danger"
    text = VOICE.get((status, reason),
                     "Danger! Stay alert!" if danger else "Warning. You look drowsy.")
    js = (ALERT_JS
          .replace("__NONCE__", str(time.time_ns()))
          .replace("__BEEP__", "true" if beep else "false")
          .replace("__BEEPS__", "2" if danger else "1")
          .replace("__FREQ__", "2200" if danger else "1200")
          .replace("__FLASH__", "true" if flash else "false")
          .replace("__COLOR__", "rgba(255,0,0,1)" if danger else "rgba(255,170,0,1)")
          .replace("__VOICE__", "true" if voice else "false")
          .replace("__TEXT__", json.dumps(text))
          .replace("__VIBRATE__", "true" if vibrate else "false")
          .replace("__VIB__", "[300,100,300,100,300]" if danger else "[250]"))
    with box:
        components.html(js, height=0)


def _bar(label, value_text, pct, color):
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="lbl"><span>{label}</span><span>{value_text}</span></div>'
            f'<div class="bar"><div style="width:{pct:.0f}%;background:{color};box-shadow:0 0 8px {color}"></div></div>')


def hud_html(attention, fatigue, perclos_pct, blink_rate, history):
    """Futuristic HUD: glowing attention ring, mini bars and a live history line."""
    color = "#22c55e" if attention >= 70 else "#f59e0b" if attention >= 40 else "#ef4444"
    r = 52
    circ = 2 * math.pi * r
    ring = (f'<svg viewBox="0 0 140 140" width="128" height="128">'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="#1f2937" stroke-width="12"/>'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{circ * attention / 100:.1f} {circ:.1f}" transform="rotate(-90 70 70)" '
            f'style="filter:drop-shadow(0 0 6px {color})"/>'
            f'<text x="70" y="75" text-anchor="middle" font-size="32" font-weight="800" fill="#f9fafb">{attention:.0f}</text>'
            f'<text x="70" y="94" text-anchor="middle" font-size="9" fill="#9ca3af" letter-spacing="2">ATTENTION</text></svg>')

    fat_col = "#22c55e" if fatigue < 0.25 else "#f59e0b" if fatigue < 0.5 else "#ef4444"
    per_col = "#22c55e" if perclos_pct < 12 else "#f59e0b" if perclos_pct < 20 else "#ef4444"
    bars = (_bar("FATIGUE", f"{fatigue * 100:.0f}%", fatigue * 100, fat_col)
            + _bar("PERCLOS", f"{perclos_pct:.0f}%", perclos_pct / 0.35, per_col)
            + _bar("BLINKS / MIN", "--" if blink_rate is None else f"{blink_rate:.0f}",
                   0 if blink_rate is None else min(100, blink_rate / 30 * 100), "#38bdf8"))

    spark = ""
    if len(history) >= 2:
        w, h = 300, 44
        n = len(history)
        pts = " ".join(f"{i / (n - 1) * w:.1f},{h - max(0, min(100, v)) / 100 * h:.1f}" for i, v in enumerate(history))
        spark = (f'<svg viewBox="0 0 {w} {h}" width="100%" height="44" preserveAspectRatio="none" style="margin-top:8px">'
                 f'<line x1="0" x2="{w}" y1="{h * 0.3:.0f}" y2="{h * 0.3:.0f}" stroke="#374151" stroke-dasharray="3 3"/>'
                 f'<line x1="0" x2="{w}" y1="{h * 0.6:.0f}" y2="{h * 0.6:.0f}" stroke="#374151" stroke-dasharray="3 3"/>'
                 f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                 f'style="filter:drop-shadow(0 0 4px {color})"/></svg>'
                 f'<div class="lbl"><span>LAST 2 MINUTES</span><span>LIVE</span></div>')
    return f'<div class="hud"><div class="hud-row">{ring}<div class="hud-bars">{bars}</div></div>{spark}</div>'


SPEAK_JS = """<script>/*__NONCE__*/
try { const W = window.parent, S = W.speechSynthesis;
  if (S && !S.speaking) { S.cancel(); S.resume();
    const u = new W.SpeechSynthesisUtterance(__TEXT__); u.lang = 'en-US'; W.__dmsCoach = u;
    W.setTimeout(function () { S.speak(u); }, 120); } } catch (e) {}
</script>"""


def speak_text(box, text):
    with box:
        components.html(SPEAK_JS.replace("__NONCE__", str(time.time_ns())).replace("__TEXT__", json.dumps(text)), height=0)


def tiles_html(items):
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tl">{label}</div><div class="tv" style="color:{color}">{value}</div></div>'
        for label, value, color in items) + '</div>'


def topbar_html(live, status, trip_time, name, ms=None):
    dot = {"Normal": "#22c55e", "Warning": "#f59e0b", "Danger": "#ef4444"}.get(status, "#22c55e") if live else "#6b7280"
    label = "LIVE" if live else "OFFLINE"
    return ('<div class="topbar"><div class="brand"><span class="logo">🚘</span><div>'
            '<div class="bt">AI DRIVER MONITOR</div><div class="bs">REAL-TIME DRIVER SAFETY · EDGE AI</div></div></div>'
            f'<div class="pills"><span class="pill"><i style="background:{dot};box-shadow:0 0 8px {dot}"></i>{label}</span>'
            f'<span class="pill">⏱ {trip_time}</span>'
            + (f'<span class="pill">⚡ {ms:.0f} ms</span>' if ms else '')
            + f'<span class="pill">🚗 {name}</span></div></div>')


STATUS_COL = {"Normal": "#7dd3fc", "Warning": "#f59e0b", "Danger": "#ef4444"}

TELLTALES = [("eyes", "👁️", "EYES"), ("yawn", "🥱", "YAWN"), ("drowsy", "😴", "DROWSY"), ("away", "↔️", "AWAY"),
             ("head", "⬇️", "HEAD"), ("phone", "📱", "PHONE"), ("eat", "🥤", "EAT"), ("face", "🙈", "FACE")]
REASON_LIGHT = {"Eyes closed": "eyes", "Yawning": "yawn", "Drowsy": "drowsy", "Drowsy (PERCLOS)": "drowsy",
                "Looking away": "away", "Head down": "head", "Using phone": "phone", "Phone call": "phone",
                "Eating or drinking": "eat", "Face not visible": "face"}


def _polar(cx, cy, r, ang):
    t = math.radians(ang)
    return cx + r * math.cos(t), cy + r * math.sin(t)


def _arc(cx, cy, r, a0, a1):
    x0, y0 = _polar(cx, cy, r, a0)
    x1, y1 = _polar(cx, cy, r, a1)
    return f"M{x0:.1f},{y0:.1f} A{r},{r} 0 {1 if a1 - a0 > 180 else 0} 1 {x1:.1f},{y1:.1f}"


def _dial(cx, cy, r, value, vmin, vmax, zones, label, readout):
    """A car-style analogue dial: tick ring, coloured zones, glowing needle, digital readout."""
    start, sweep = 135.0, 270.0
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    o = [f'<circle cx="{cx}" cy="{cy}" r="{r + 8}" fill="url(#dialbg)" stroke="#1e293b" stroke-width="1.5"/>',
         f'<path d="{_arc(cx, cy, r, start, start + sweep)}" fill="none" stroke="#0b1220" stroke-width="8" stroke-linecap="round"/>']
    for z0, z1, col in zones:
        a0 = start + sweep * (z0 - vmin) / (vmax - vmin)
        a1 = start + sweep * (z1 - vmin) / (vmax - vmin)
        o.append(f'<path d="{_arc(cx, cy, r, a0, a1)}" fill="none" stroke="{col}" stroke-width="4" opacity="0.85"/>')
    for i in range(41):
        ang = start + sweep * i / 40
        major = i % 8 == 0
        x1, y1 = _polar(cx, cy, r - 6, ang)
        x2, y2 = _polar(cx, cy, r - (16 if major else 11), ang)
        o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{"#e2e8f0" if major else "#475569"}" stroke-width="{1.8 if major else 1}"/>')
    for k in range(6):
        xv, yv = _polar(cx, cy, r - 27, start + sweep * k / 5)
        o.append(f'<text x="{xv:.1f}" y="{yv + 3:.1f}" text-anchor="middle" font-size="8.5" fill="#94a3b8">'
                 f'{vmin + (vmax - vmin) * k / 5:.0f}</text>')
    ang = start + sweep * frac
    tx, ty = _polar(cx, cy, r - 10, ang)
    bx, by = _polar(cx, cy, -12, ang)
    o.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#ef4444" stroke-width="2.6" '
             f'stroke-linecap="round" filter="url(#glow)"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="#0b1220" stroke="#ef4444" stroke-width="1.5"/>')
    o.append(f'<text x="{cx}" y="{cy + r * 0.52:.1f}" text-anchor="middle" font-size="20" font-weight="700" fill="#f8fafc">{readout}</text>')
    o.append(f'<text x="{cx}" y="{cy + r * 0.52 + 13:.1f}" text-anchor="middle" font-size="7.5" letter-spacing="2" fill="#64748b">{label}</text>')
    return "".join(o)


def cluster_html(live, status, reason, attention, fatigue, perclos_pct, blink_rate, n_alerts,
                 trip_time, head, history):
    """Luxury-car style instrument cluster."""
    col = STATUS_COL.get(status, "#7dd3fc") if live else "#64748b"
    green, amber, red = "#22c55e", "#f59e0b", "#ef4444"
    defs = ('<defs><radialGradient id="dialbg" cx="50%" cy="45%" r="60%"><stop offset="0%" stop-color="#111827"/>'
            '<stop offset="100%" stop-color="#020617"/></radialGradient>'
            '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
    gem = {"Normal": "✓", "Warning": "!", "Danger": "✕"}.get(status, "✓") if live else "•"
    svg = (f'<svg viewBox="0 0 460 190" width="100%" xmlns="http://www.w3.org/2000/svg">{defs}'
           + _dial(108, 94, 76, attention if live else 0, 0, 100, [(0, 40, red), (40, 70, amber), (70, 100, green)],
                   "ATTENTION", f"{attention:.0f}" if live else "--")
           + _dial(352, 94, 76, (fatigue * 100) if live else 0, 0, 100, [(0, 25, green), (25, 50, amber), (50, 100, red)],
                   "FATIGUE %", f"{fatigue * 100:.0f}" if live else "--")
           + f'<circle cx="230" cy="94" r="27" fill="#05080f" stroke="{col}" stroke-width="2" filter="url(#glow)"/>'
           + f'<text x="230" y="104" text-anchor="middle" font-size="28" font-weight="700" fill="{col}">{gem}</text>'
           + '</svg>')

    if not live:
        text = "SYSTEM STANDBY"
    elif reason == "Calibrating":
        text = "CALIBRATING · LOOK AT THE ROAD"
    elif status == "Normal":
        text = "ATTENTIVE"
    else:
        text = f"{status.upper()} · {reason.upper()}"
    lit = REASON_LIGHT.get(reason) if (live and status != "Normal") else None
    tell = "".join(
        f'<div class="tl-i {"" if key == lit else "tl-off"}" '
        f'style="{"filter:drop-shadow(0 0 6px " + col + ")" if key == lit else ""}">{ico}<small>{name}</small></div>'
        for key, ico, name in TELLTALES)

    spark = ""
    if live and len(history) >= 2:
        w, h, n = 300, 26, len(history)
        pts = " ".join(f"{i / (n - 1) * w:.1f},{h - max(0, min(100, v)) / 100 * h:.1f}" for i, v in enumerate(history))
        spark = (f'<svg viewBox="0 0 {w} {h}" width="100%" height="26" preserveAspectRatio="none" style="margin-top:6px">'
                 f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6" '
                 f'style="filter:drop-shadow(0 0 3px {col})"/></svg>')
    br = "--" if blink_rate is None else f"{blink_rate:.0f}"
    cells = [("TRIP", trip_time), ("ALERTS", str(n_alerts)), ("BLINKS/MIN", br), ("PERCLOS", f"{perclos_pct:.0f}%"),
             ("HEAD", f"{head[0]:+.0f}° {head[1]:+.0f}°")]
    trip = '<div class="trip">' + "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in cells) + "</div>"
    cls = "cluster cl-danger" if (live and status == "Danger") else "cluster"
    return (f'<div class="{cls}" style="border-color:{col}55">{svg}'
            f'<div class="cl-status" style="color:{col}">{text}</div><div class="tell">{tell}</div>{spark}{trip}</div>')


def render_report(driver_name):
    """Trip report shown after the camera is stopped."""
    import pandas as pd
    trip = st.session_state.get("trip")
    if trip is None or len(trip.samples) < 3:
        st.info("📊 Your trip report will appear here after you stop the camera.")
        return
    s = trip.summary()
    st.subheader("📊 Trip report")
    items = [("Safety score", f"{s['score']}/100 ({s['grade']})"), ("Trip time", fmt_time(s["duration"])),
             ("Unattentive", f"{s['unattentive_pct']:.0f}%"), ("Warnings / Dangers", f"{s['warnings']} / {s['dangers']}"),
             ("Max fatigue", f"{s['max_fatigue']:.2f}"), ("Max PERCLOS", f"{s['max_perclos'] * 100:.0f}%"),
             ("Phone use", f"{int(s['phone_s'])} s"), ("Eating / drinking", f"{int(s['eating_s'])} s")]
    for row in (items[:4], items[4:]):
        for col, (k, v) in zip(st.columns(4), row):
            col.metric(k, v)

    df = pd.DataFrame(trip.samples)
    df["Minutes"] = df["t"] / 60.0
    st.line_chart(df.rename(columns={"fatigue": "Fatigue", "perclos": "PERCLOS"})
                    .set_index("Minutes")[["Fatigue", "PERCLOS"]])

    if s["events"]:
        st.caption("Alerts during the trip")
        st.dataframe(pd.DataFrame([{"At": fmt_time(ev["start"]),
                                    "Length (s)": max(1, int(ev["end"] - ev["start"])),
                                    "Level": ev["level"], "Reason": ev["reason"]}
                                   for ev in s["events"]]), hide_index=True)
    else:
        st.success("No alerts during this trip. Great driving!")

    st.download_button("⬇️ Download report (HTML)", build_report_html(trip, driver_name),
                       file_name="trip_report.html", mime="text/html")
    st.download_button("⬇️ Download data (CSV)", samples_csv(trip),
                       file_name="trip_data.csv", mime="text/csv")


topbar_box = st.empty()
topbar_box.markdown(topbar_html(False, "Normal", "00:00", "My vehicle"), unsafe_allow_html=True)

left, right = st.columns([9, 11])

with left:
    ctx = webrtc_streamer(
        key="driver-monitor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DriverProcessor,
        media_stream_constraints={
            "video": {"facingMode": "user", "width": {"ideal": 320}, "height": {"ideal": 240},
                      "frameRate": {"ideal": 12, "max": 15}},
            "audio": False,
        },
        video_html_attrs={"style": {"width": "100%", "borderRadius": "14px",
                                    "border": "1px solid rgba(56,189,248,0.35)",
                                    "boxShadow": "0 0 20px rgba(56,189,248,0.18)"},
                          "controls": False, "autoPlay": True, "playsInline": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    alerts_on = st.session_state.get("alerts_on", True)
    if st.button("🔔  ALERTS: ON" if alerts_on else "🔕  ALERTS: OFF", key="alerts_btn",
                 type="primary" if alerts_on else "secondary"):
        st.session_state["alerts_on"] = not alerts_on
        st.rerun()
    bc1, bc2, bc3 = st.columns(3)
    btn_reset = bc1.button("🔄 Reset")
    btn_cal = bc2.button("🎯 Calibrate")
    btn_test = bc3.button("🔊 Test")
    calib_box = st.empty()
    notify_box = st.empty()
    sound_box = st.empty()
    test_box = st.empty()
    geo_box = st.empty()
    coach_sound_box = st.empty()

with right:
    cluster_box = st.empty()
    coach_box = st.empty()
    loc_box = st.empty()

tab_owner, tab_report = st.tabs(["📨 Owner", "📊 Trip report"])
opt_beep = opt_voice = opt_flash = opt_vibrate = alerts_on

with tab_owner:
    o1, o2 = st.columns(2)
    notify_on = o1.toggle("Message the owner if driver stays inattentive", value=True)
    if HAS_GEO:
        loc_on = o2.toggle("📍 Include location in alerts", value=False)
    else:
        loc_on = False
        o2.caption("📍 Location needs streamlit-js-eval (see requirements.txt).")
    o3, o4 = st.columns(2)
    notify_after = o3.number_input("Send after (seconds)", min_value=10, max_value=600, value=60, step=5)
    driver_name = o4.text_input("Driver / vehicle name", value="My vehicle")
    if st.button("📨 Send test message"):
        ok, info = send_telegram(f"✅ Test message from AI Driver Monitor ({driver_name}). Notifications are working.")
        st.success("Test message sent!") if ok else st.error(info)

report_slot = tab_report.empty()
report_slot.info("📊 Your trip report will appear here after you stop the camera.")

# ---- button actions (they need the settings above)
if btn_reset and ctx.video_processor:
    ctx.video_processor.monitor.reset()
if btn_cal:
    if ctx.video_processor:
        ctx.video_processor.monitor.calib.start(time.time())
    else:
        st.warning("Tap START on the camera first.")
if btn_test:
    fire_alert(test_box, "Danger", "Eyes closed", opt_beep, opt_voice, opt_flash, opt_vibrate)

# ---- phone location (refreshed about every 30 seconds)
geo_window = int(time.time() // 30)
if loc_on and HAS_GEO:
    with geo_box:
        try:
            g = get_geolocation(component_key=f"geo_{geo_window}")
        except Exception:
            g = None
    if isinstance(g, dict) and isinstance(g.get("coords"), dict):
        co = g["coords"]
        if co.get("latitude") is not None and co.get("longitude") is not None:
            st.session_state["loc"] = (co["latitude"], co["longitude"], time.time())


if ctx.state.playing:
    last_sig = last_top = last_coach_sig = None
    while ctx.state.playing:
        p = ctx.video_processor
        if p is not None:
            m = p.monitor
            now = time.time()
            st.session_state["trip"] = m.trip

            n_alerts = len(m.trip.events) + (1 if m.trip.cur else 0)
            trip_time = fmt_time(now - (m.trip.start or now))
            if st.session_state.get("trip") is not m.trip:
                st.session_state["trip"] = m.trip

            # redraw only when something visibly changed (keeps the page light)
            br = m.blink_rate(now)
            sig = (m.status, m.reason, int(m.attention), int(m.fatigue * 100), int(m.perclos_pct), n_alerts,
                   round(m.yaw_rel / 2), round(m.pitch_rel / 2), None if br is None else int(br), trip_time)
            if sig != last_sig:
                last_sig = sig
                hist = [x.get("attention", 100) for x in m.trip.samples[-120:]]
                cluster_box.markdown(cluster_html(True, m.status, m.reason, m.attention, m.fatigue, m.perclos_pct,
                                                  br, n_alerts, trip_time, (m.yaw_rel, m.pitch_rel), hist),
                                     unsafe_allow_html=True)
            top_sig = (m.status, trip_time, int(p.proc_ms // 10))
            if top_sig != last_top:
                last_top = top_sig
                topbar_box.markdown(topbar_html(True, m.status, trip_time, driver_name, p.proc_ms),
                                    unsafe_allow_html=True)

            lvl, advice = m.coach(now)
            if (lvl, advice) != last_coach_sig:
                last_coach_sig = (lvl, advice)
                coach_box.markdown(f'<div class="coach coach-{lvl}">🧠 <b>Co-pilot:</b> {advice}</div>',
                                   unsafe_allow_html=True)
            if lvl == "break" and opt_voice and now - st.session_state.get("last_coach", 0) > 300:
                speak_text(coach_sound_box, advice)
                st.session_state["last_coach"] = now

            # ---- calibration status
            cal = m.calib
            if cal.active:
                calib_box.progress(cal.progress(now), text="🎯 Calibrating: look at the road ahead as you normally drive, eyes open")
            elif cal.failed:
                calib_box.warning(cal.message)
            elif cal.done:
                calib_box.caption("🎯 " + cal.message)

            # ---- location status
            loc = st.session_state.get("loc")
            if loc_on and loc:
                loc_box.caption(f"📍 Location updated {int(now - loc[2])} s ago")

            # ---- owner notification after continuous inattention
            if notify_on:
                if m.bad_seconds > 0:
                    notify_box.progress(min(m.bad_seconds / notify_after, 1.0),
                                        text=f"Inattentive for {int(m.bad_seconds)}s / {int(notify_after)}s")
                else:
                    notify_box.empty()
                if m.bad_seconds >= notify_after and not st.session_state.get("notified") \
                        and now - st.session_state.get("last_sent", 0) > 120:
                    jpg = None
                    if p.last_img is not None:
                        okj, buf = cv2.imencode(".jpg", p.last_img)
                        jpg = buf.tobytes() if okj else None
                    where = ""
                    if loc_on and loc:
                        where = f"Location: https://maps.google.com/?q={loc[0]},{loc[1]} (updated {int(now - loc[2])} s ago)\n"
                    msg = (f"🚨 DRIVER ALERT - {driver_name}\n"
                           f"The driver has been inattentive for {int(m.bad_seconds)} seconds.\n"
                           f"Reason: {m.reason or 'Drowsy'}\n"
                           f"{where}"
                           f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC\n"
                           f"Please call the driver immediately.")
                    ok, info = send_telegram(msg, jpg)
                    if ok and loc_on and loc:
                        send_telegram_location(loc[0], loc[1])
                    st.session_state["notified"] = True
                    st.session_state["last_sent"] = now
                    st.toast("📨 Owner notified" if ok else f"⚠️ Notification failed: {info}")
                if m.bad_seconds == 0:
                    st.session_state["notified"] = False

            # ---- phone alerts (beep + voice + flash + vibration)
            if opt_voice:
                gap = 2.5 if m.status == "Danger" else 4.0
            else:
                gap = 1.4 if m.status == "Danger" else 2.0
            last_beep = st.session_state.get("last_beep", 0.0)
            if m.status in ("Warning", "Danger") and now - last_beep > gap:
                fire_alert(sound_box, m.status, m.reason, opt_beep, opt_voice, opt_flash, opt_vibrate)
                st.session_state["last_beep"] = now
            elif m.status == "Normal" and now - last_beep > 2.0:
                sound_box.empty()

            # ---- refresh the location component every 30 s
            if loc_on and HAS_GEO and int(now // 30) != geo_window:
                st.rerun()
        time.sleep(0.4)

# camera is stopped (or was never started): show the report
if not ctx.state.playing:
    cluster_box.markdown(cluster_html(False, "Normal", "", 0, 0, 0, None, 0, "00:00", (0, 0), []), unsafe_allow_html=True)
    notify_box.empty()
    topbar_box.markdown(topbar_html(False, "Normal", "00:00", driver_name), unsafe_allow_html=True)
    with report_slot.container():
        render_report(driver_name)
