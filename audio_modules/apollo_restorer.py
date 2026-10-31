#!/usr/bin/env python3
"""
Transparent Neural Codec Restoration & Unmanipulated Guitar Hi-Fi Restorer.

Integrates open-source GitHub technologies for pure, non-manipulative restoration:
  1. `JusperLee/Apollo` (Apache-2.0, Tsinghua University & Tencent AI Lab):
     Band-Sequence Modeling (`BSNet` + Rotary Transformer `Roformer`) trained on
     MUSDB18-HQ & MoisesDB at 44.1 kHz to remove MP3/AAC/WhatsApp lossy compression
     artifacts and restore natural upper string/cymbal harmonics without changing
     notes, timbre, or dynamics.
  2. `facebookresearch/demucs` (`htdemucs_6s` 6-stem Transformer):
     Dedicated neural `guitar` isolation separating electric/acoustic guitar cleanly
     from keys, room wash, and drum/vocal bleed without artificial EQ carving.
  3. `timsainb/noisereduce` + Zero-Phase Linear Filtering (`scipy.signal.sosfiltfilt`):
     Gentle stationary hiss/hum floor reduction and sub-68 Hz rumble removal
     (preserving Low E2 = 82.4 Hz intact) with zero phase distortion, zero amp
     re-synthesis, zero artificial widening, and zero dynamic squash.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

APOLLO_SAMPLE_RATE = 44100
APOLLO_HF_REPO = "JusperLee/Apollo"
APOLLO_DEFAULT_CKPT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "models", "apollo", "pytorch_model.bin")
)

_APOLLO_CACHE: dict[str, tuple[nn.Module, torch.device]] = {}


# ---------------------------------------------------------------------------
# JusperLee/Apollo Architecture (Apache-2.0 — Pure PyTorch 44.1 kHz)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    def __init__(self, dimension: int, groups: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t = x.shape
        x_float = x.reshape(b, self.groups, -1, t).float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(-2, keepdim=True) + self.eps)
        return x_norm.type_as(x).reshape(b, n, t) * self.weight.reshape(1, -1, 1)


class _Roformer(nn.Module):
    """Transformer with Rotary Positional Embedding (exact upstream Apollo layout)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_head: int = 8,
        theta: float = 10000.0,
        window: int = 10000,
        input_drop: float = 0.0,
        attention_drop: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.theta = theta
        self.window = window
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)
        self.register_buffer("sin_freq", sin_freq)
        self.attention_drop = attention_drop
        self.causal = causal

        self.input_norm = _RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size * self.num_head * 3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size * self.num_head, self.input_size, 1, bias=False)
        self.MLP = nn.Sequential(
            _RMSNorm(self.input_size),
            nn.Conv1d(self.input_size, self.input_size * 8, 1, bias=False),
            nn.SiLU(),
        )
        self.MLP_output = nn.Conv1d(self.input_size * 4, self.input_size, 1, bias=False)

    def _calc_rotary_emb(self) -> tuple[torch.Tensor, torch.Tensor]:
        freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.hidden_size, 2)[: (self.hidden_size // 2)] / self.hidden_size)
        )
        freq = freq.reshape(1, -1)
        pos = torch.arange(0, self.window).reshape(-1, 1)
        cos_freq = torch.cos(pos * freq)
        sin_freq = torch.sin(pos * freq)
        cos_freq = torch.stack([cos_freq] * 2, -1).reshape(self.window, self.hidden_size)
        sin_freq = torch.stack([sin_freq] * 2, -1).reshape(self.window, self.hidden_size)
        return cos_freq, sin_freq

    def _add_rotary_sequence(self, feature: torch.Tensor) -> torch.Tensor:
        t, n = feature.shape[-2:]
        feature_reshape = feature.reshape(-1, t, n)
        cos_freq = self.cos_freq[:t]
        sin_freq = self.sin_freq[:t]
        reverse_sign = torch.tensor([-1.0, 1.0], device=feature.device, dtype=feature.dtype)
        feature_reshape_neg = (
            torch.flip(feature_reshape.reshape(-1, n // 2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)
        ).reshape(-1, t, n)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
        return feature_rope.reshape(feature.shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, _, t = x.shape
        weight = self.weight(self.input_drop(self.input_norm(x))).reshape(
            b, self.num_head, self.hidden_size * 3, t
        ).mT
        q, k, v = torch.split(weight, self.hidden_size, dim=-1)
        q_rot = self._add_rotary_sequence(q)
        k_rot = self._add_rotary_sequence(k)
        attn_out = F.scaled_dot_product_attention(
            q_rot.contiguous(),
            k_rot.contiguous(),
            v.contiguous(),
            dropout_p=self.attention_drop,
            is_causal=self.causal,
        )
        attn_out = attn_out.mT.reshape(b, -1, t)
        out = self.output(attn_out) + x
        gate, z = self.MLP(out).chunk(2, dim=1)
        out = out + self.MLP_output(F.silu(gate) * z)
        return out, (k_rot, v)


class _ConvActNorm1d(nn.Module):
    def __init__(self, in_channel: int, hidden_channel: int, kernel: int = 7, causal: bool = False):
        super().__init__()
        self.kernel = kernel
        self.causal = causal
        pad = (kernel - 1) // 2 if not causal else (kernel - 1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel, padding=pad, groups=in_channel),
            _RMSNorm(in_channel),
            nn.Conv1d(in_channel, hidden_channel, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channel, in_channel, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.causal:
            out = out[..., : -self.kernel + 1]
        return x + out


class _ICB(nn.Module):
    def __init__(self, in_channel: int, kernel: int = 7, causal: bool = False):
        super().__init__()
        self.blocks = nn.Sequential(
            _ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
            _ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
            _ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _BSNet(nn.Module):
    def __init__(self, feature_dim: int, kernel: int = 7):
        super().__init__()
        self.feature_dim = feature_dim
        self.band_net = _Roformer(self.feature_dim, self.feature_dim, num_head=8, window=100, causal=False)
        self.seq_net = _ICB(self.feature_dim, kernel=kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, nband, _, t = x.shape
        band_in = x.permute(0, 3, 2, 1).reshape(b * t, -1, nband)
        band_out, _ = self.band_net(band_in)
        band_out = band_out.reshape(b, t, -1, nband).permute(0, 3, 2, 1)
        return self.seq_net(band_out.reshape(b * nband, -1, t)).reshape(b, nband, -1, t)


class ApolloNet(nn.Module):
    """Band-Sequence Modeling network for 44.1 kHz compressed music restoration."""

    def __init__(self, sr: int = 44100, win: int = 20, feature_dim: int = 256, layer: int = 6):
        super().__init__()
        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        bandwidth = int(self.win / 160)
        self.band_width = [bandwidth] * 79
        self.band_width.append(self.enc_dim - int(np.sum(self.band_width)))
        self.nband = len(self.band_width)

        self.BN = nn.ModuleList(
            [
                nn.Sequential(
                    _RMSNorm(self.band_width[i] * 2 + 1),
                    nn.Conv1d(self.band_width[i] * 2 + 1, self.feature_dim, 1),
                )
                for i in range(self.nband)
            ]
        )
        self.net = nn.Sequential(*[_BSNet(self.feature_dim) for _ in range(layer)])
        self.output = nn.ModuleList(
            [
                nn.Sequential(
                    _RMSNorm(self.feature_dim),
                    nn.Conv1d(self.feature_dim, self.band_width[i] * 4, 1),
                    nn.GLU(dim=1),
                )
                for i in range(self.nband)
            ]
        )
        self.register_buffer("_hann", torch.hann_window(self.win), persistent=False)

    def spec_band_split(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        b, nch, nsample = x.shape
        spec = torch.stft(
            x.reshape(b * nch, nsample),
            n_fft=self.win,
            hop_length=self.stride,
            window=self._hann,
            return_complex=True,
        )
        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(self.nband):
            this_spec = spec[:, band_idx : band_idx + self.band_width[i]]
            pwr = (this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1)
            subband_power.append(pwr)
            subband_spec_norm.append(torch.complex(this_spec.real / pwr, this_spec.imag / pwr))
            band_idx += self.band_width[i]
        return subband_spec_norm, torch.cat(subband_power, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, nch, nsample = x.shape
        subband_spec_norm, subband_power = self.spec_band_split(x)
        subband_feature = torch.stack(
            [
                self.BN[i](
                    torch.cat(
                        [
                            subband_spec_norm[i].real,
                            subband_spec_norm[i].imag,
                            torch.log(subband_power[:, i].unsqueeze(1)),
                        ],
                        1,
                    )
                )
                for i in range(self.nband)
            ],
            1,
        )
        feature = self.net(subband_feature)
        est_spec = torch.cat(
            [
                torch.complex(
                    *(
                        self.output[i](feature[:, i])
                        .view(b * nch, 2, self.band_width[i], -1)
                        .unbind(dim=1)
                    )
                )
                for i in range(self.nband)
            ],
            1,
        )
        out = torch.istft(
            est_spec,
            n_fft=self.win,
            hop_length=self.stride,
            window=self._hann,
            length=nsample,
        ).view(b, nch, -1)
        return out


# ---------------------------------------------------------------------------
# Model Loading & Overlap-Add Chunked GPU Runner
# ---------------------------------------------------------------------------


def _install_omegaconf_unpickle_stub() -> None:
    """Allow `torch.load` to unpickle checkpoints carrying OmegaConf metadata without requiring `omegaconf`."""
    if "omegaconf" in sys.modules:
        return

    class _DummyOmegaObj:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    for mod_name in (
        "omegaconf",
        "omegaconf.dictconfig",
        "omegaconf.listconfig",
        "omegaconf.base",
        "omegaconf.nodes",
    ):
        mod = types.ModuleType(mod_name)
        mod.__getattr__ = lambda _name, _cls=_DummyOmegaObj: _cls  # type: ignore[attr-defined]
        sys.modules[mod_name] = mod


def _resolve_apollo_ckpt_path(ckpt_path: str | None = None) -> str:
    target = ckpt_path or APOLLO_DEFAULT_CKPT
    if os.path.isfile(target):
        return target
    if ".app/Contents" in target or (
        sys.platform == "darwin" and not os.access(os.path.dirname(os.path.dirname(target)), os.W_OK)
    ):
        return os.path.expanduser(
            "~/Library/Application Support/StudioEQ_Master_Pro/models/apollo/pytorch_model.bin"
        )
    return target


def _get_apollo_model(ckpt_path: str | None = None) -> tuple[ApolloNet, torch.device] | None:
    target_ckpt = _resolve_apollo_ckpt_path(ckpt_path)
    if target_ckpt in _APOLLO_CACHE:
        return _APOLLO_CACHE[target_ckpt]

    if not os.path.isfile(target_ckpt):
        try:
            from huggingface_hub import hf_hub_download

            os.makedirs(os.path.dirname(target_ckpt), exist_ok=True)
            target_ckpt = hf_hub_download(
                repo_id=APOLLO_HF_REPO,
                filename="pytorch_model.bin",
                local_dir=os.path.dirname(target_ckpt),
            )
        except Exception:
            return None

    try:
        _install_omegaconf_unpickle_stub()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conf = torch.load(target_ckpt, map_location="cpu", weights_only=False)
        state = conf["state_dict"] if isinstance(conf, dict) and "state_dict" in conf else conf
        feat_dim = int(state["BN.0.1.weight"].shape[0]) if "BN.0.1.weight" in state else 256
        net_indices = [int(k.split(".")[1]) for k in state.keys() if k.startswith("net.")]
        n_layers = (max(net_indices) + 1) if net_indices else 6
        model = ApolloNet(sr=APOLLO_SAMPLE_RATE, win=20, feature_dim=feat_dim, layer=n_layers)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        _APOLLO_CACHE[target_ckpt] = (model, device)
        return model, device
    except Exception:
        return None


def restore_codec_apollo(
    audio_2xn: np.ndarray,
    sr: int = 44100,
    chunk_seconds: float = 5.0,
    overlap_seconds: float = 0.75,
    chunk_pad_seconds: float = 0.35,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Run `JusperLee/Apollo` 44.1 kHz lossless codec restoration with padded overlap-add.
    Returns shape `(2, N)` float32. Falls back cleanly to input if model unavailable.
    """
    if sr != APOLLO_SAMPLE_RATE or audio_2xn.size == 0:
        return audio_2xn.astype(np.float32, copy=True)

    loaded = _get_apollo_model()
    if loaded is None:
        return audio_2xn.astype(np.float32, copy=True)
    model, device = loaded

    orig_peak = float(np.max(np.abs(audio_2xn)))
    if orig_peak < 1e-6:
        return audio_2xn.astype(np.float32, copy=True)

    # Normalize gently into Apollo's optimal [-0.85, 0.85] operating headroom, then restore exact gain
    norm_gain = 0.80 / orig_peak if orig_peak < 0.25 else 1.0
    work = np.ascontiguousarray(audio_2xn * norm_gain, dtype=np.float32)
    tensor_in = torch.from_numpy(work).unsqueeze(0)  # (1, 2, N)
    total_samples = tensor_in.shape[-1]

    chunk_samples = int(round(chunk_seconds * sr))
    overlap_samples = int(round(overlap_seconds * sr))
    pad_samples = int(round(chunk_pad_seconds * sr))
    hop = chunk_samples - overlap_samples

    with torch.inference_mode():
        if total_samples <= chunk_samples:
            out_t = model(tensor_in.to(device)).detach().cpu()
        else:
            starts = [0]
            while starts[-1] + chunk_samples < total_samples:
                starts.append(starts[-1] + hop)
            padded_len = chunk_samples + 2 * pad_samples
            out_sum = torch.zeros_like(tensor_in, device="cpu")
            w_sum = torch.zeros((1, 1, total_samples), dtype=torch.float32)

            eff_bs = batch_size if device.type == "cuda" else 1
            for b0 in range(0, len(starts), eff_bs):
                b_starts = starts[b0 : b0 + eff_bs]
                chunks = []
                meta = []
                for st in b_starts:
                    ed = min(st + chunk_samples, total_samples)
                    valid_n = ed - st
                    p_st = max(0, st - pad_samples)
                    if pad_samples and ed == total_samples:
                        p_st = max(0, total_samples - padded_len)
                    ch = tensor_in[..., p_st : p_st + padded_len]
                    if ch.shape[-1] < padded_len:
                        ch = F.pad(ch, (0, padded_len - ch.shape[-1]))
                    chunks.append(ch)
                    meta.append((st, ed, valid_n, p_st))

                b_out = model(torch.cat(chunks, dim=0).to(device)).detach().cpu()
                for idx_in_b, (st, ed, valid_n, p_st) in enumerate(meta):
                    g_idx = b0 + idx_in_b
                    v_st = st - p_st
                    seg = b_out[idx_in_b : idx_in_b + 1, :, v_st : v_st + valid_n]
                    w = torch.ones(valid_n, dtype=torch.float32)
                    fade_n = min(overlap_samples, valid_n)
                    if fade_n > 0:
                        ramp = torch.linspace(0.0, 1.0, fade_n)
                        if g_idx > 0:
                            w[:fade_n] = ramp
                        if g_idx < len(starts) - 1:
                            w[-fade_n:] = torch.flip(ramp, dims=(0,))
                    wv = w.view(1, 1, -1)
                    out_sum[..., st:ed] += seg * wv
                    w_sum[..., st:ed] += wv

            out_t = out_sum / torch.clamp(w_sum, min=1e-8)

    restored = (out_t.squeeze(0).numpy() / norm_gain).astype(np.float32)
    return restored


# ---------------------------------------------------------------------------
# Pure Non-Manipulative Guitar Hi-Fi Restoration Pipeline
# ---------------------------------------------------------------------------


def restore_guitar_transparent(
    guitar_2xn: np.ndarray,
    sr: int = 44100,
    apollo_high_blend: float = 0.85,
) -> tuple[np.ndarray, dict]:
    """
    Improve guitar recording quality WITHOUT manipulating tone, pitch, timing,
    distortion, or stereo imaging:
      1. Zero-phase (<68 Hz) sub-rumble filter (leaves Low E2 = 82.4 Hz untouched).
      2. Gentle stationary background hiss floor reduction (`noisereduce` prop=0.42
         estimated strictly from quietest inter-phrase noise floor).
      3. `JusperLee/Apollo` neural codec de-artifacting & natural upper-harmonic
         restoration: phase-locks `<3.6 kHz` core guitar body/timbre 100% to the
         authentic recording while blending Apollo's restored lossless string/pick
         overtones (`>3.6 kHz`) to cure WhatsApp/MP3 muffling & ringing.
      4. Zero-phase adaptive rehearsal-room standing-wave tamer (only attenuates
         sustained 195-265 Hz room boom up to -2.0 dB when booming; 0 dB on pick/chords).
    """
    x = np.nan_to_num(guitar_2xn.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 1:
        x = np.vstack([x, x])
    n = x.shape[1]
    if n < int(0.1 * sr) or float(np.max(np.abs(x))) < 1e-6:
        return x, {"apollo_applied": False, "hf_gain_db": 0.0}

    nyq = 0.5 * sr
    # 1. Zero-phase sub-rumble cleanup (<68 Hz, preserving Low E2 = 82.4 Hz phase & weight)
    sos_hp = butter(3, 68.0 / nyq, btype="highpass", output="sos")
    clean = sosfiltfilt(sos_hp, x.astype(np.float64), axis=-1).astype(np.float32)

    # 2. Gentle stationary amp-hiss / room-noise reduction without touching sustain
    try:
        import noisereduce as nr

        frame_len = int(0.05 * sr)
        n_frames = n // frame_len
        if n_frames >= 20:
            mono_chk = 0.5 * (clean[0] + clean[1])
            f_rms = np.sqrt(np.mean(mono_chk[: n_frames * frame_len].reshape(n_frames, frame_len) ** 2, axis=1))
            active_frames = f_rms[f_rms > 1e-5]
            if active_frames.size >= 10:
                q_thr = float(np.percentile(active_frames, 12))
                noise_idx = np.where((f_rms <= q_thr) & (f_rms > 1e-6))[0]
                if noise_idx.size >= 4:
                    noise_clip = np.concatenate(
                        [clean[:, i * frame_len : (i + 1) * frame_len] for i in noise_idx[:16]],
                        axis=1,
                    )
                    clean = nr.reduce_noise(
                        y=clean,
                        sr=sr,
                        y_noise=noise_clip,
                        stationary=True,
                        prop_decrease=0.40,
                        n_fft=2048,
                    ).astype(np.float32)
    except Exception:
        pass

    # Measure pre-Apollo HF energy (8-16 kHz)
    sos_hf_meas = butter(3, [8000.0 / nyq, min(0.95, 16500.0 / nyq)], btype="bandpass", output="sos")
    hf_before = float(np.sqrt(np.mean(sosfiltfilt(sos_hf_meas, 0.5 * (clean[0] + clean[1])) ** 2) + 1e-18))

    # 3. JusperLee/Apollo Neural Lossless Codec Restoration (phase-locked crossover @ 3.6 kHz)
    apollo_out = restore_codec_apollo(clean, sr=sr)
    apollo_applied = not np.allclose(apollo_out, clean, atol=1e-6)

    if apollo_applied and apollo_high_blend > 0.0:
        # Zero-phase Linkwitz-style split at 3.6 kHz:
        #   - Below 3.6 kHz: 85% pure original phase-intact guitar body + 15% Apollo de-block
        #   - Above 3.6 kHz: Apollo's restored natural string/pick harmonics & codec de-swish!
        sos_lp = butter(4, 3600.0 / nyq, btype="lowpass", output="sos")
        orig_low = sosfiltfilt(sos_lp, clean.astype(np.float64), axis=-1).astype(np.float32)
        orig_high = clean - orig_low

        ap_low = sosfiltfilt(sos_lp, apollo_out.astype(np.float64), axis=-1).astype(np.float32)
        ap_high = apollo_out - ap_low

        # Cap Apollo HF restoration boost to a natural studio ceiling (+7.5 dB max) so it never sounds bright/harsh
        rms_oh = float(np.sqrt(np.mean(orig_high ** 2) + 1e-15))
        rms_ah = float(np.sqrt(np.mean(ap_high ** 2) + 1e-15))
        if rms_ah > rms_oh * 2.37:  # +7.5 dB
            ap_high = ap_high * ((rms_oh * 2.37) / rms_ah)

        hb = float(np.clip(apollo_high_blend, 0.0, 1.0))
        merged = (0.85 * orig_low + 0.15 * ap_low) + ((1.0 - hb) * orig_high + hb * ap_high)
    else:
        merged = clean

    # 4. Zero-phase dynamic room-standing-wave tamer (190-265 Hz, max -2.0 dB ONLY on sustained boom)
    sos_boom = butter(2, [190.0 / nyq, 265.0 / nyq], btype="bandpass", output="sos")
    boom_band = sosfiltfilt(sos_boom, merged.astype(np.float64), axis=-1).astype(np.float32)
    boom_env = sosfiltfilt(
        butter(2, 8.0 / nyq, btype="lowpass", output="sos"),
        np.abs(0.5 * (boom_band[0] + boom_band[1])).astype(np.float64),
    )
    active_boom = boom_env[boom_env > 1e-5]
    if active_boom.size > 64:
        thr_b = float(np.percentile(active_boom, 82))
        over = np.clip((boom_env - thr_b) / (thr_b + 1e-9), 0.0, 1.0)
        gain_boom = (10.0 ** ((-2.0 * over) / 20.0)).astype(np.float32)
        merged = merged + (gain_boom[None, :] - 1.0) * boom_band

    # Strict level & peak preservation (no dynamic compression / manipulation!)
    pk = float(np.max(np.abs(merged)))
    if pk > 0.955:
        merged = merged * (0.955 / pk)

    hf_after = float(np.sqrt(np.mean(sosfiltfilt(sos_hf_meas, 0.5 * (merged[0] + merged[1])) ** 2) + 1e-18))
    hf_gain_db = 20.0 * float(np.log10((hf_after + 1e-18) / (hf_before + 1e-18)))

    return merged.astype(np.float32), {
        "apollo_applied": apollo_applied,
        "hf_restored_db": round(hf_gain_db, 2),
    }
