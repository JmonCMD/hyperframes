"""Synthesize the temp score for "La casa de naipes" (60 s, deterministic).

Cues follow the script's sound notes:
  0-27 s   rising music bed (pad + pulse + riser), hard cut at 27.0
  27-33 s  low grey hum under "la grieta"
  33.0 s   dry impact, then 1 s of silence
  34-47 s  tense low drone
  47-54 s  bright shimmer for the AI scene, gone by 54
  54-60 s  no music (voice only, then silence)

Usage: python3 -I gen-audio.py <out.wav>
"""

import sys
import wave

import numpy as np

SR = 48000
DUR = 60.0
t = np.arange(int(SR * DUR)) / SR
rng = np.random.default_rng(2008)
out = np.zeros_like(t)


def window(start, end, fade_in=0.05, fade_out=0.01):
    """Gain envelope that is 1 inside [start, end] with short linear fades."""
    g = np.clip((t - start) / max(fade_in, 1e-4), 0, 1)
    g *= np.clip((end - t) / max(fade_out, 1e-4), 0, 1)
    return g


def lowpass(x, alpha):
    """One-pole lowpass (alpha in 0..1, smaller = darker)."""
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


# --- 0-27: euphoric bed. A major-ish pad whose level and brightness climb. ---
bed = np.zeros_like(t)
for n in (45, 52, 57, 61, 64):  # A2 E3 A3 C#4 E4
    f = hz(n)
    bed += np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t + 0.3)
# second chord (F#m) from 12 s and (D) from 19 s for movement
for n, s, e in ((54, 12, 19), (57, 12, 19), (50, 19, 27), (62, 19, 27), (66, 19, 27)):
    f = hz(n)
    bed += window(s, e, 0.6, 0.2) * 0.8 * np.sin(2 * np.pi * f * t)
climb = np.clip(t / 27.0, 0, 1)
bed *= (0.18 + 0.55 * climb**1.6) * window(0.0, 27.0, 1.2, 0.004)
bed /= 6.0

# Pulse: soft kick on every beat (100 bpm) from 5 s, hats join at 12 s.
beat = 60 / 100
pulse = np.zeros_like(t)
k = 5.0
while k < 27.0 - 0.05:
    i0 = int(k * SR)
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    kick = np.sin(2 * np.pi * (48 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
    end = min(i0 + n, len(t))
    pulse[i0:end] += 0.55 * kick[: end - i0] * (0.4 + 0.6 * k / 27)
    if k >= 12.0:
        h0 = int((k + beat / 2) * SR)
        hn = int(0.06 * SR)
        if h0 + hn < int(27.0 * SR):
            hat = rng.standard_normal(hn) * np.exp(-np.arange(hn) / SR * 70)
            pulse[h0 : h0 + hn] += 0.08 * np.diff(np.concatenate([[0], hat]))
    k += beat

# Riser: filtered noise that opens up from 19 to 27 s.
noise = rng.standard_normal(len(t))
riser = (noise - lowpass(noise, 0.08)) * window(19.0, 27.0, 7.5, 0.004)
riser *= np.clip((t - 19) / 8, 0, 1) ** 2 * 0.12

out += bed + pulse + riser

# --- 27-33: low grey hum ---
hum = (np.sin(2 * np.pi * 55 * t) + 0.4 * np.sin(2 * np.pi * 82.4 * t)) * 0.07
out += hum * window(27.25, 32.6, 0.8, 0.4)

# --- 33.0: dry impact (sub boom + noise crack), short decay, then silence ---
i0 = int(33.0 * SR)
n = int(0.9 * SR)
tt = np.arange(n) / SR
boom = np.sin(2 * np.pi * (38 + 70 * np.exp(-tt * 18)) * tt) * np.exp(-tt * 5.5)
crack = rng.standard_normal(n) * np.exp(-tt * 40)
hit = 0.9 * boom + 0.35 * crack
hit *= np.clip((0.9 - tt) / 0.15, 0, 1)
out[i0 : i0 + n] += hit

# --- 34-47: tense low drone (D minor, slow tremolo) ---
drone = np.zeros_like(t)
for n_ in (38, 45, 53):  # D2 A2 F3
    drone += np.sin(2 * np.pi * hz(n_) * t)
drone *= (0.75 + 0.25 * np.sin(2 * np.pi * 0.5 * t)) * 0.05
out += drone * window(34.2, 47.0, 1.5, 0.6)

# --- 47-54: bright shimmer for the AI scene ---
shimmer = np.zeros_like(t)
for n_ in (57, 64, 69, 73, 76):  # A3 E4 A4 C#5 E5
    shimmer += np.sin(2 * np.pi * hz(n_) * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3.1 * t + n_))
out += shimmer / 5 * 0.16 * window(47.0, 54.0, 1.0, 1.2)

# Master: hard silence from 54 s on (voice only, then black).
out *= window(0.0, 54.0, 0.0001, 0.3)
out = np.tanh(out * 1.4) * 0.6
stereo = np.stack([out, out], axis=1)
pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)

with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
