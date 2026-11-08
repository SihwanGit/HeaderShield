import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from data.dataset import get_dataloaders
from models.header_bank import HeaderBank
from utils.audio import MelSpectrogramConverter
from utils.header_selector import RuleBasedHeaderSelector
from utils.header_training_log import HeaderTrainingLog, slice_metadata


# ============================================================
# Encoder option defaults
# AntiFake run.py와 비슷하게 사용자가 loss encoder를 선택할 수 있도록 분리한다.
# 현재 이 파일에서 실제 학습 연결까지 구현된 것은 DUMMY_LOSS와 RTVC_LOSS다.
# AVC / COQUI / TORTOISE는 추후 adapter 추가를 위한 옵션 자리만 먼저 만들어 둔다.
# ============================================================

DUMMY_LOSS_DEFAULT = False
RTVC_LOSS_DEFAULT = True
AVC_LOSS_DEFAULT = False
COQUI_LOSS_DEFAULT = False
TORTOISE_AUTOREGRESSIVE_LOSS_DEFAULT = False
TORTOISE_DIFFUSION_LOSS_DEFAULT = False


def str2bool(value: Any) -> bool:
    """
    argparse에서 True/False 문자열을 안정적으로 받기 위한 helper.
    예:
        --rtvc_loss true
        --dummy_loss false
    """
    if isinstance(value, bool):
        return value

    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def move_converter_to_device(converter: MelSpectrogramConverter, device: str) -> MelSpectrogramConverter:
    """
    MelSpectrogramConverter 내부 torchaudio transform들을 현재 device로 이동한다.
    GPU 환경에서 waveform은 cuda에 있는데 mel_transform은 cpu에 남아 있으면 device mismatch가 발생할 수 있다.
    """
    if hasattr(converter, "to"):
        return converter.to(device)

    converter.mel_transform = converter.mel_transform.to(device)
    converter.inverse_mel_transform = converter.inverse_mel_transform.to(device)
    converter.griffin_lim = converter.griffin_lim.to(device)
    return converter


def match_mel_shapes(source_mel: torch.Tensor, target_mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    source_mel과 target_mel의 주파수/시간 shape를 공통 크기로 맞춘다.

    Args:
        source_mel: 원본 화자 mel [1, 1, F, T]
        target_mel: 목표 화자 mel [1, 1, F, T]

    Returns:
        shape가 맞춰진 source_mel, target_mel
    """
    freq_len = min(source_mel.shape[2], target_mel.shape[2])
    time_len = min(source_mel.shape[3], target_mel.shape[3])

    source_mel = source_mel[:, :, :freq_len, :time_len]
    target_mel = target_mel[:, :, :freq_len, :time_len]

    return source_mel, target_mel


def crop_to_header_length(source_mel: torch.Tensor,
                          target_mel: torch.Tensor,
                          header_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Header Bank 학습은 발화 시작부 보호가 목적이므로 mel의 앞부분 header_length만 사용한다.

    Args:
        source_mel: 원본 화자 mel [1, 1, F, T]
        target_mel: 목표 화자 mel [1, 1, F, T]
        header_length: 사용할 시작 구간 길이. mel frame 단위

    Returns:
        앞부분만 잘라낸 source_mel, target_mel
    """
    source_mel, target_mel = match_mel_shapes(source_mel, target_mel)

    time_len = min(source_mel.shape[-1], target_mel.shape[-1], header_length)

    source_mel = source_mel[:, :, :, :time_len]
    target_mel = target_mel[:, :, :, :time_len]

    return source_mel, target_mel


def freeze_model(model: nn.Module) -> None:
    """
    speaker_encoder는 학습하지 않고 Header Bank만 학습하므로 encoder parameter를 고정한다.
    RTVC adapter처럼 내부 모델을 lazy-load하는 경우도 있으므로 parameters()가 비어 있어도 허용한다.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    selector feature 값이 tensor/numpy/float 중 어떤 형태로 와도 로그에 출력 가능한 float으로 변환한다.
    """
    try:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)
    except Exception:
        return default


def normalize_header_index(header_index: int, num_headers: int) -> int:
    """
    selector가 반환한 header index를 HeaderBank 범위 안으로 보정한다.
    """
    if header_index < 0:
        return 0
    if header_index >= num_headers:
        return num_headers - 1
    return header_index


def ensure_2d_waveform(wav: torch.Tensor) -> torch.Tensor:
    """
    waveform shape를 [1, T]로 맞춘다.
    """
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.dim() == 2:
        if wav.size(0) == 1:
            return wav
        # multi-channel이면 첫 채널만 사용
        return wav[:1, :]
    if wav.dim() == 3:
        # [B, C, T] 형태가 들어오면 첫 batch/첫 channel 사용
        return wav[0, :1, :]
    raise ValueError(f"Unsupported waveform shape: {tuple(wav.shape)}")


def mel_to_waveform_compat(converter: MelSpectrogramConverter,
                           mel: torch.Tensor,
                           device: str) -> torch.Tensor:
    """
    MelSpectrogramConverter 구현체의 메서드명이 다를 수 있으므로 여러 후보를 순서대로 시도한다.

    기대 입력:
        mel: [1, 1, F, T] 또는 [1, F, T] 또는 [F, T]

    반환:
        waveform: [1, T]

    주의:
        RTVC 연결은 mel-domain header를 waveform-domain encoder로 전달해야 하므로
        mel -> waveform 복원이 필요하다. 현재 프로젝트의 converter가 Griffin-Lim 기반
        역변환을 제공한다고 가정한다.
    """
    x = mel

    # converter가 [1, F, T] 또는 [F, T]를 기대할 수 있으므로 후보를 준비한다.
    candidates = []
    candidates.append(x)
    if x.dim() == 4 and x.size(0) == 1:
        candidates.append(x.squeeze(0))       # [1, F, T]
        candidates.append(x.squeeze(0).squeeze(0))  # [F, T]
    elif x.dim() == 3 and x.size(0) == 1:
        candidates.append(x.squeeze(0))       # [F, T]

    method_names = [
        "mel_to_waveform",
        "mel_to_audio",
        "mel_to_wav",
        "inverse",
        "mel_to_wave",
    ]

    last_error = None

    for method_name in method_names:
        if not hasattr(converter, method_name):
            continue

        method = getattr(converter, method_name)
        for cand in candidates:
            try:
                wav = method(cand)
                wav = ensure_2d_waveform(wav)
                return wav.to(device)
            except Exception as exc:
                last_error = exc

    # fallback: converter 내부 transform을 직접 사용한다.
    # 이 fallback은 converter가 inverse_mel_transform과 griffin_lim을 public attribute로 갖는 경우에만 동작한다.
    try:
        cand = x
        if cand.dim() == 4:
            cand = cand.squeeze(0).squeeze(0)  # [F, T]
        elif cand.dim() == 3:
            cand = cand.squeeze(0)             # [F, T]

        linear_spec = converter.inverse_mel_transform(cand)
        wav = converter.griffin_lim(linear_spec)
        wav = ensure_2d_waveform(wav)
        return wav.to(device)
    except Exception as exc:
        last_error = exc

    raise RuntimeError(
        "mel을 waveform으로 복원할 수 없습니다.\n"
        "RTVC encoder는 waveform-domain 입력을 사용하므로 MelSpectrogramConverter에 "
        "mel_to_waveform/mel_to_audio 등의 역변환 메서드가 필요합니다.\n"
        f"last_error={repr(last_error)}"
    )


class DummySpeakerEncoder(nn.Module):
    """
    기존 smoke test용 dummy speaker encoder.
    실제 보호 성능을 주장하기 위한 모델이 아니라 코드 경로 검증용이다.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 128)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RTVCSpeakerEncoderAdapter(nn.Module):
    """
    AntiFake에서 분리해 온 SV2TTS / RTVC speaker encoder adapter.

    예상 폴더 구조:
        VSMask/
        ├── external/
        │   ├── rtvc/
        │   │   ├── encoder/
        │   │   └── utils/
        │   └── saved_models/
        │       └── default/
        │           └── encoder.pt

    사용 방식:
        - sys.path에 external/rtvc를 추가
        - external/saved_models/default/encoder.pt 로드
        - source waveform에서 mel_slices를 계산
        - protected waveform을 같은 mel_slices 기준으로 embedding화
    """
    def __init__(self,
                 external_root: str,
                 sample_rate: int,
                 device: str,
                 threshold: float,
                 loss_scale: float):
        super().__init__()

        self.external_root = Path(external_root).resolve()
        self.rtvc_root = self.external_root / "rtvc"
        self.model_root = self.external_root / "saved_models"
        self.model_path = self.model_root / "default" / "encoder.pt"

        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.loss_scale = float(loss_scale)

        if not self.rtvc_root.exists():
            raise FileNotFoundError(f"RTVC root not found: {self.rtvc_root}")

        if not self.model_path.exists():
            raise FileNotFoundError(
                "RTVC encoder checkpoint가 없습니다.\n"
                f"expected: {self.model_path}\n"
                "현재 구조는 external/rtvc 와 external/saved_models/default/encoder.pt 를 가정합니다."
            )

        if str(self.rtvc_root) not in sys.path:
            sys.path.insert(0, str(self.rtvc_root))

        from encoder import inference as rtvc_inference
        from encoder import audio as rtvc_audio

        self.rtvc_inference = rtvc_inference
        self.rtvc_audio = rtvc_audio

        # external/saved_models/default/encoder.pt가 이미 존재한다고 가정하고 직접 로드한다.
        # VSMask 프로젝트의 utils 패키지와 RTVC의 utils 패키지 이름이 충돌하므로
        # utils.default_models.ensure_default_models는 사용하지 않는다.
        self.rtvc_inference.load_model(self.model_path)

        self.encoder_model = None

    def _prepare_slices_and_model(self, wav_tensor: torch.Tensor):
        """
        source waveform에서 RTVC partial mel_slices와 encoder model 객체를 준비한다.
        이 과정은 numpy 전처리를 사용하므로 source 기준 정보 준비용으로만 사용한다.
        """
        wav_tensor = ensure_2d_waveform(wav_tensor.detach()).to("cpu")
        wav_np = wav_tensor.squeeze(0).numpy().astype(np.float32)

        preprocessed_wav = self.rtvc_inference.preprocess_wav(wav_np, self.sample_rate)

        wav_np_processed, _, mel_slices, encoder_model, _ = self.rtvc_inference.embed_utterance_preprocess(
            preprocessed_wav,
            using_partials=True
        )

        self.encoder_model = encoder_model.to(self.device)
        self.encoder_model.train()

        for param in self.encoder_model.parameters():
            param.requires_grad = False

        processed_tensor = torch.from_numpy(wav_np_processed).float().unsqueeze(0).to(self.device)

        return processed_tensor, mel_slices

    def embed_waveform_with_slices(self,
                                   wav_tensor: torch.Tensor,
                                   mel_slices) -> torch.Tensor:
        """
        torch waveform tensor에서 RTVC embedding을 계산한다.
        이 함수는 protected waveform에 대해 gradient가 유지되어야 하므로 numpy 변환을 사용하지 않는다.
        """
        if self.encoder_model is None:
            raise RuntimeError("RTVC encoder_model이 준비되지 않았습니다. _prepare_slices_and_model을 먼저 호출해야 합니다.")

        wav_tensor = ensure_2d_waveform(wav_tensor).to(self.device)

        frames_tensor = self.rtvc_audio.wav_to_mel_spectrogram_torch(wav_tensor).to(self.device)

        embeds = []
        for s in mel_slices:
            try:
                frame_tensor = frames_tensor[s].unsqueeze(0).to(self.device)
                embed = self.encoder_model.forward(frame_tensor)
                embeds.append(embed)
            except Exception:
                continue

        if len(embeds) == 0:
            raise RuntimeError(
                "RTVC embedding 계산에 사용할 수 있는 mel slice가 없습니다. "
                "header_length/window_size가 너무 짧을 가능성이 있습니다."
            )

        partial_embeds = torch.stack(embeds, dim=0)
        raw_embed = torch.mean(partial_embeds, dim=0, keepdim=True)

        return raw_embed

    def compute_threshold_loss_from_mel(self,
                                        source_mel: torch.Tensor,
                                        target_mel: torch.Tensor,
                                        perturbed_mel: torch.Tensor,
                                        converter: MelSpectrogramConverter,
                                        args) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        mel-domain HeaderBank 출력을 waveform으로 복원한 뒤 RTVC encoder loss를 계산한다.

        본 연구 코드에서는 AntiFake의 target-based/threshold-based 분기 옵션을 사용하지 않는다.
        HeaderBank의 목적은 특정 target speaker로 이동하는 것이 아니라,
        protected embedding이 source embedding에서 일정 threshold 이상 멀어지도록 만드는 것이다.

        따라서 RTVC loss는 항상 다음 방식으로 고정한다.

            loss = ELU(rtvc_threshold - ||E(protected) - E(source)||_2) * rtvc_scale

        이 loss는 protected embedding과 source embedding의 거리가 threshold보다 작을 때 penalty를 주고,
        threshold 이상 멀어지면 penalty가 작아지도록 동작한다.
        """
        source_wav = mel_to_waveform_compat(converter, source_mel, args.device)
        protected_wav = mel_to_waveform_compat(converter, perturbed_mel, args.device)

        # source 기준 mel_slices 준비
        processed_source_wav, mel_slices = self._prepare_slices_and_model(source_wav)

        with torch.no_grad():
            source_embedding = self.embed_waveform_with_slices(processed_source_wav, mel_slices).detach()

        protected_embedding = self.embed_waveform_with_slices(protected_wav, mel_slices)

        distance_source = torch.norm(protected_embedding - source_embedding, p=2)

        threshold = torch.tensor(self.threshold, device=self.device)
        elu = torch.nn.ELU()
        speaker_loss = elu(threshold - distance_source) * self.loss_scale

        logs = {
            "speaker": float(speaker_loss.detach().cpu().item()),
            # target speaker 방향으로 끌고 가는 loss를 사용하지 않으므로 target은 0.0으로 둔다.
            "target": 0.0,
            "source": float(distance_source.detach().cpu().item()),
            "rtvc_distance_source": float(distance_source.detach().cpu().item()),
            "rtvc_threshold": float(self.threshold)
        }

        return speaker_loss, logs


class SpeakerLossManager(nn.Module):
    """
    여러 encoder loss를 선택적으로 묶기 위한 manager.

    현재 구현:
        - DUMMY_LOSS
        - RTVC_LOSS

    추후 확장 예정:
        - AVC_LOSS
        - COQUI_LOSS
        - TORTOISE_AUTOREGRESSIVE_LOSS
        - TORTOISE_DIFFUSION_LOSS
    """
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dummy_encoder: Optional[DummySpeakerEncoder] = None
        self.rtvc_encoder: Optional[RTVCSpeakerEncoderAdapter] = None

        if args.dummy_loss:
            self.dummy_encoder = DummySpeakerEncoder().to(args.device)

        if args.rtvc_loss:
            self.rtvc_encoder = RTVCSpeakerEncoderAdapter(
                external_root=args.external_root,
                sample_rate=args.sample_rate,
                device=args.device,
                threshold=args.rtvc_threshold,
                loss_scale=args.rtvc_scale
            )

        unsupported = []
        if args.avc_loss:
            unsupported.append("AVC_LOSS")
        if args.coqui_loss:
            unsupported.append("COQUI_LOSS")
        if args.tortoise_autoregressive_loss:
            unsupported.append("TORTOISE_AUTOREGRESSIVE_LOSS")
        if args.tortoise_diffusion_loss:
            unsupported.append("TORTOISE_DIFFUSION_LOSS")

        if unsupported:
            raise NotImplementedError(
                "현재 train_header_bank_condition_subset.py에는 다음 encoder adapter가 아직 구현되지 않았습니다: "
                + ", ".join(unsupported)
                + "\n우선 --rtvc_loss true --dummy_loss false 조합으로 RTVC 연결 테스트를 진행하세요."
            )

        if self.dummy_encoder is None and self.rtvc_encoder is None:
            raise ValueError(
                "사용할 speaker encoder loss가 없습니다. "
                "--dummy_loss true 또는 --rtvc_loss true 중 하나 이상을 켜세요."
            )

    def freeze_all(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def compute_loss(self,
                     source_mel: torch.Tensor,
                     target_mel: torch.Tensor,
                     perturbed_mel: torch.Tensor,
                     converter: MelSpectrogramConverter,
                     args) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: List[torch.Tensor] = []
        logs: Dict[str, float] = {}

        if self.dummy_encoder is not None:
            dummy_loss, dummy_logs = compute_dummy_speaker_embedding_loss(
                source_mel=source_mel,
                target_mel=target_mel,
                perturbed_mel=perturbed_mel,
                speaker_encoder=self.dummy_encoder,
                lambda_param=args.lambda_param
            )
            losses.append(dummy_loss * args.dummy_scale)

            for key, value in dummy_logs.items():
                logs[f"dummy_{key}"] = value

        if self.rtvc_encoder is not None:
            rtvc_loss, rtvc_logs = self.rtvc_encoder.compute_threshold_loss_from_mel(
                source_mel=source_mel,
                target_mel=target_mel,
                perturbed_mel=perturbed_mel,
                converter=converter,
                args=args
            )
            losses.append(rtvc_loss)

            for key, value in rtvc_logs.items():
                logs[f"rtvc_{key}"] = value

        if len(losses) == 0:
            raise RuntimeError("Speaker loss를 계산할 encoder가 없습니다.")

        total_speaker_loss = torch.stack(losses).sum()

        logs["speaker"] = float(total_speaker_loss.detach().cpu().item())

        # 기존 로그 키 호환성을 위해 target/source는 대표값을 둔다.
        if "dummy_target" in logs:
            logs["target"] = logs["dummy_target"]
            logs["source"] = logs["dummy_source"]
        elif "rtvc_target" in logs:
            logs["target"] = logs["rtvc_target"]
            logs["source"] = logs["rtvc_source"]
        else:
            logs["target"] = 0.0
            logs["source"] = 0.0

        return total_speaker_loss, logs


def compute_dummy_speaker_embedding_loss(source_mel: torch.Tensor,
                                         target_mel: torch.Tensor,
                                         perturbed_mel: torch.Tensor,
                                         speaker_encoder: nn.Module,
                                         lambda_param: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    기존 Header Bank의 dummy speaker embedding loss를 계산한다.

    목표:
        1. perturbed embedding이 target speaker embedding에 가까워지게 함
        2. perturbed embedding이 source speaker embedding에서 멀어지게 함
    """
    with torch.no_grad():
        source_embedding = speaker_encoder(source_mel).detach()
        target_embedding = speaker_encoder(target_mel).detach()

    perturbed_embedding = speaker_encoder(perturbed_mel)

    loss_target = F.mse_loss(perturbed_embedding, target_embedding)
    loss_source = F.mse_loss(perturbed_embedding, source_embedding)

    speaker_loss = loss_target - lambda_param * loss_source

    logs = {
        "speaker": float(speaker_loss.detach().cpu().item()),
        "target": float(loss_target.detach().cpu().item()),
        "source": float(loss_source.detach().cpu().item())
    }

    return speaker_loss, logs


def compute_temporal_smoothness(diff_mel: torch.Tensor) -> torch.Tensor:
    """
    mel-domain perturbation이 시간축에서 지나치게 튀지 않도록 하는 smoothness regularization.
    실제 STFT loss는 waveform 복원이 필요하므로, 이 파일 안에서는 mel-domain temporal smoothness를 surrogate로 사용한다.
    """
    if diff_mel.shape[-1] <= 1:
        return diff_mel.new_tensor(0.0)

    return (diff_mel[..., 1:] - diff_mel[..., :-1]).pow(2).mean()


def compute_high_frequency_penalty(diff_mel: torch.Tensor,
                                   high_freq_start_ratio: float) -> torch.Tensor:
    """
    고주파 mel bin 영역에서 perturbation이 과도하게 커지는 것을 억제한다.
    Header 3처럼 fricative/high-frequency 시작 구간에서 청각적 왜곡을 줄이기 위해 사용한다.
    """
    mel_bins = diff_mel.shape[2]
    start_bin = int(mel_bins * high_freq_start_ratio)
    start_bin = max(0, min(start_bin, mel_bins - 1))

    return diff_mel[:, :, start_bin:, :].pow(2).mean()


def compute_onset_speaker_loss(source_mel: torch.Tensor,
                               target_mel: torch.Tensor,
                               perturbed_mel: torch.Tensor,
                               speaker_loss_manager: SpeakerLossManager,
                               converter: MelSpectrogramConverter,
                               args,
                               onset_frames: int) -> torch.Tensor:
    """
    발화 시작부 앞쪽 일부 frame만 잘라 speaker loss를 한 번 더 계산한다.
    Header 2의 abrupt/high-energy onset 공격성을 강화하기 위해 사용한다.
    """
    time_len = min(source_mel.shape[-1], target_mel.shape[-1], perturbed_mel.shape[-1], onset_frames)

    if time_len <= 0:
        return source_mel.new_tensor(0.0)

    onset_source = source_mel[..., :time_len]
    onset_target = target_mel[..., :time_len]
    onset_perturbed = perturbed_mel[..., :time_len]

    onset_loss, _ = speaker_loss_manager.compute_loss(
        source_mel=onset_source,
        target_mel=onset_target,
        perturbed_mel=onset_perturbed,
        converter=converter,
        args=args
    )

    return onset_loss


def compute_noise_consistency_loss(source_mel: torch.Tensor,
                                   perturbed_mel: torch.Tensor,
                                   header_bank: HeaderBank,
                                   header_index: int,
                                   speaker_loss_manager: SpeakerLossManager,
                                   converter: MelSpectrogramConverter,
                                   args,
                                   noise_std: float) -> torch.Tensor:
    """
    Header 4의 noise-robust consistency loss.
    source_mel에 작은 잡음을 더한 뒤 같은 header를 적용했을 때,
    깨끗한 입력에서의 perturbed speaker loss와 noisy 입력에서의 speaker loss가
    크게 달라지지 않도록 약한 consistency를 둔다.

    RTVC처럼 waveform 복원이 필요한 encoder도 있으므로 embedding 직접 MSE 대신
    speaker loss 값의 차이를 사용하는 형태로 단순화했다.
    """
    if noise_std <= 0:
        return source_mel.new_tensor(0.0)

    noisy_source = source_mel + torch.randn_like(source_mel) * noise_std
    noisy_perturbed = header_bank.apply_header(noisy_source, header_index)

    clean_loss, _ = speaker_loss_manager.compute_loss(
        source_mel=source_mel,
        target_mel=source_mel,
        perturbed_mel=perturbed_mel,
        converter=converter,
        args=args
    )

    noisy_loss, _ = speaker_loss_manager.compute_loss(
        source_mel=noisy_source,
        target_mel=source_mel,
        perturbed_mel=noisy_perturbed,
        converter=converter,
        args=args
    )

    return F.mse_loss(noisy_loss, clean_loss.detach())


def compute_condition_aware_header_loss(source_mel: torch.Tensor,
                                        target_mel: torch.Tensor,
                                        header_bank: HeaderBank,
                                        header_index: int,
                                        speaker_loss_manager: SpeakerLossManager,
                                        converter: MelSpectrogramConverter,
                                        args) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    header index별로 다른 regularization/constraint를 적용한다.

    Header 0:
        fallback 안정성을 위해 L2 + temporal smoothness regularization을 적용한다.

    Header 1:
        저에너지 시작 구간에서는 작은 잡음도 잘 들릴 수 있으므로 perturbation norm penalty를 더 강하게 적용한다.

    Header 2:
        abrupt/high-energy onset에서 speaker feature 교란을 강화하기 위해 speaker loss와 onset speaker loss를 증폭한다.

    Header 3:
        high spectral centroid/ZCR 구간에서 고주파 왜곡을 줄이기 위해 high-frequency penalty를 적용한다.

    Header 4:
        noisy onset에 대응하기 위해 noise consistency loss를 적용한다.
    """
    perturbed_mel = header_bank.apply_header(source_mel, header_index)

    speaker_loss, logs = speaker_loss_manager.compute_loss(
        source_mel=source_mel,
        target_mel=target_mel,
        perturbed_mel=perturbed_mel,
        converter=converter,
        args=args
    )

    diff_mel = perturbed_mel - source_mel
    l2_loss = diff_mel.pow(2).mean()
    smooth_loss = compute_temporal_smoothness(diff_mel)
    high_freq_loss = compute_high_frequency_penalty(
        diff_mel=diff_mel,
        high_freq_start_ratio=args.h3_high_freq_start_ratio
    )

    total_loss = speaker_loss

    logs["l2"] = float(l2_loss.detach().cpu().item())
    logs["smooth"] = float(smooth_loss.detach().cpu().item())
    logs["high_freq"] = float(high_freq_loss.detach().cpu().item())
    logs["onset"] = 0.0
    logs["consistency"] = 0.0

    if header_index == 0:
        # Header 0: 기본 fallback이므로 평균적인 품질 안정성을 중시한다.
        total_loss = (
            speaker_loss
            + args.h0_l2_weight * l2_loss
            + args.h0_smooth_weight * smooth_loss
        )

    elif header_index == 1:
        # Header 1: 저에너지 발화는 잡음이 더 잘 느껴질 수 있으므로 norm penalty를 강하게 둔다.
        total_loss = (
            speaker_loss
            + args.h1_norm_weight * l2_loss
            + args.h1_smooth_weight * smooth_loss
        )

    elif header_index == 2:
        # Header 2: 갑작스러운 시작부는 onset 구간의 speaker 교란을 강화한다.
        onset_loss = compute_onset_speaker_loss(
            source_mel=source_mel,
            target_mel=target_mel,
            perturbed_mel=perturbed_mel,
            speaker_loss_manager=speaker_loss_manager,
            converter=converter,
            args=args,
            onset_frames=args.h2_onset_frames
        )
        logs["onset"] = float(onset_loss.detach().cpu().item())

        total_loss = (
            args.h2_attack_boost * speaker_loss
            + args.h2_onset_attack_weight * onset_loss
            + args.h2_l2_weight * l2_loss
        )

    elif header_index == 3:
        # Header 3: 고주파/마찰음 구간에서는 high-frequency perturbation을 억제한다.
        total_loss = (
            speaker_loss
            + args.h3_high_freq_weight * high_freq_loss
            + args.h3_l2_weight * l2_loss
        )

    elif header_index == 4:
        # Header 4: 잡음성 시작 구간에서는 noise consistency를 추가한다.
        consistency_loss = compute_noise_consistency_loss(
            source_mel=source_mel,
            perturbed_mel=perturbed_mel,
            header_bank=header_bank,
            header_index=header_index,
            speaker_loss_manager=speaker_loss_manager,
            converter=converter,
            args=args,
            noise_std=args.h4_noise_std
        )
        logs["consistency"] = float(consistency_loss.detach().cpu().item())

        total_loss = (
            speaker_loss
            + args.h4_consistency_weight * consistency_loss
            + args.h4_l2_weight * l2_loss
        )

    else:
        # num_headers가 5보다 클 경우를 위한 기본 처리
        total_loss = speaker_loss + args.extra_header_l2_weight * l2_loss

    logs["total"] = float(total_loss.detach().cpu().item())

    return total_loss, logs


def build_condition_subsets(train_loader,
                            selector: RuleBasedHeaderSelector,
                            args, training_log=None) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[int, int]]:
    """
    학습 전에 전체 데이터셋의 발화 시작 구간을 분석하고 header 조건별 subset을 만든다.

    파이프라인:
        데이터셋의 각 발화 시작 구간 분석
        → feature 기반으로 조건 분류
        → 조건별 데이터 subset 구성
        → 각 subset으로 해당 header를 따로 학습
    """
    subsets: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(args.num_headers)}
    all_examples: List[Dict[str, Any]] = []
    selected_counts: Dict[int, int] = {idx: 0 for idx in range(args.num_headers)}

    collected = 0

    print("[1단계] 데이터셋 발화 시작 구간 분석 및 조건별 subset 구성")

    for batch in tqdm.tqdm(train_loader, desc="Building condition subsets"):
        source_waveform = batch["source_waveform"]
        target_waveform = batch["target_waveform"]

        for i in range(source_waveform.size(0)):
            if args.max_subset_samples > 0 and collected >= args.max_subset_samples:
                break

            single_source_waveform = source_waveform[i].detach().cpu().clone()
            single_target_waveform = target_waveform[i].detach().cpu().clone()

            header_index, features = selector.select(
                single_source_waveform,
                return_features=True
            )
            raw_header_index = header_index
            header_index = normalize_header_index(header_index, args.num_headers)
            metadata = slice_metadata(batch, i, args.sample_rate)
            if training_log is not None:
                training_log.assignment(metadata, raw_header_index, header_index, features)

            normalized_features = {
                "energy": safe_float(features.get("energy", 0.0)),
                "zero_crossing_rate": safe_float(features.get("zero_crossing_rate", 0.0)),
                "spectral_centroid": safe_float(features.get("spectral_centroid", 0.0))
            }

            example = {
                "source_waveform": single_source_waveform,
                "target_waveform": single_target_waveform,
                "assigned_header": header_index,
                "metadata": metadata,
                "features": normalized_features
            }

            subsets[header_index].append(example)
            all_examples.append(example)
            selected_counts[header_index] += 1
            collected += 1

        if args.max_subset_samples > 0 and collected >= args.max_subset_samples:
            break

    if len(all_examples) == 0:
        raise RuntimeError(
            "조건별 subset을 구성할 수 없습니다. 데이터셋이 비어 있거나 dataloader가 샘플을 반환하지 않았습니다."
        )

    print("[Subset 구성 결과]")
    for idx in range(args.num_headers):
        print(f"  header {idx}: {len(subsets[idx])} samples")

    if training_log is not None:
        training_log.summary("subset-built")
    return subsets, all_examples, selected_counts


def build_examples_for_header(header_index: int,
                              subsets: Dict[int, List[Dict[str, Any]]],
                              all_examples: List[Dict[str, Any]],
                              args) -> List[Tuple[Dict[str, Any], float, str]]:
    """
    특정 header를 학습하기 위한 example 목록을 만든다.

    기본 원칙:
        해당 header에 배정된 subset을 primary data로 사용한다.

    데이터 부족 시:
        subset 크기가 min_subset_size보다 작으면 borrowed data를 낮은 가중치로 보충한다.
        borrowed_sample_weight는 default_header_weight보다 클 수 없도록 main에서 검증한다.

    Returns:
        [(example, sample_weight, sample_type), ...]
    """
    primary_examples = subsets.get(header_index, [])
    training_examples: List[Tuple[Dict[str, Any], float, str]] = []

    for ex in primary_examples:
        training_examples.append((ex, args.primary_sample_weight, "primary"))

    if len(primary_examples) < args.min_subset_size and args.enable_borrowed_samples:
        shortage = args.min_subset_size - len(primary_examples)

        borrowed_candidates = [
            ex for ex in all_examples
            if ex["assigned_header"] != header_index
        ]

        if len(borrowed_candidates) == 0:
            borrowed_candidates = all_examples

        borrowed_iter = cycle(borrowed_candidates)

        for _ in range(shortage):
            borrowed_ex = next(borrowed_iter)
            training_examples.append((borrowed_ex, args.borrowed_sample_weight, "borrowed"))

    return training_examples


def train_header_bank(speaker_loss_manager: SpeakerLossManager, args) -> None:
    """
    조건별 subset 기반 Header Bank 학습.

    수정된 핵심 구조:
        1. 전체 학습 데이터를 먼저 selector로 분석한다.
        2. 발화 시작 조건에 따라 subset[0~4]로 분리한다.
        3. 각 header는 자기 subset을 primary data로 사용해 따로 학습한다.
        4. subset이 부족한 header는 borrowed data를 낮은 가중치로 보충한다.
        5. default header 0은 fallback 안정성을 위해 선택적으로 보조 업데이트한다.
        6. header별 condition-aware loss를 적용한다.

    Header index 의미:
        0: default
        1: low-energy start
        2: high-energy / abrupt start
        3: high spectral centroid / fricative-like start
        4: noisy-like start
    """
    print("조건별 subset 기반 Header Bank 학습을 시작합니다...")

    print("[설정 확인]")
    print(f"  data_dir               : {args.data_dir}")
    print(f"  target_speaker         : {args.target_speaker}")
    print(f"  other_speakers         : {args.other_speakers}")
    print(f"  num_headers            : {args.num_headers}")
    print(f"  sample_rate            : {args.sample_rate}")
    print(f"  window_size            : {args.window_size} samples ({args.window_size / args.sample_rate:.3f} sec)")
    print(f"  shift_size             : {args.shift_size} samples ({args.shift_size / args.sample_rate:.3f} sec)")
    print(f"  hop_length             : {args.hop_length} samples")
    print(f"  header_length          : {args.header_length} mel frames ({args.header_length * args.hop_length / args.sample_rate:.3f} sec)")
    print(f"  device                 : {args.device}")

    print("[Encoder/Loss 설정]")
    print(f"  dummy_loss             : {args.dummy_loss}")
    print(f"  rtvc_loss              : {args.rtvc_loss}")
    print(f"  avc_loss               : {args.avc_loss}")
    print(f"  coqui_loss             : {args.coqui_loss}")
    print(f"  tortoise_ar_loss       : {args.tortoise_autoregressive_loss}")
    print(f"  tortoise_diff_loss     : {args.tortoise_diffusion_loss}")
    print(f"  external_root          : {args.external_root}")
    print(f"  rtvc_threshold         : {args.rtvc_threshold}")
    print(f"  rtvc_scale             : {args.rtvc_scale}")
    print(f"  dummy_scale            : {args.dummy_scale}")

    print("[학습 가중치 확인]")
    print(f"  primary_sample_weight  : {args.primary_sample_weight}")
    print(f"  default_header_weight  : {args.default_header_weight}")
    print(f"  borrowed_sample_weight : {args.borrowed_sample_weight}")
    print(f"  min_subset_size        : {args.min_subset_size}")
    print(f"  enable_borrowed_samples: {args.enable_borrowed_samples}")
    print(f"  enable_default_update  : {args.enable_default_update}")

    print("[Header별 추가 loss/제약]")
    print(f"  header 0: h0_l2_weight={args.h0_l2_weight}, h0_smooth_weight={args.h0_smooth_weight}")
    print(f"  header 1: h1_norm_weight={args.h1_norm_weight}, h1_smooth_weight={args.h1_smooth_weight}")
    print(f"  header 2: h2_attack_boost={args.h2_attack_boost}, h2_onset_attack_weight={args.h2_onset_attack_weight}, h2_onset_frames={args.h2_onset_frames}, h2_l2_weight={args.h2_l2_weight}")
    print(f"  header 3: h3_high_freq_weight={args.h3_high_freq_weight}, h3_high_freq_start_ratio={args.h3_high_freq_start_ratio}, h3_l2_weight={args.h3_l2_weight}")
    print(f"  header 4: h4_consistency_weight={args.h4_consistency_weight}, h4_noise_std={args.h4_noise_std}, h4_l2_weight={args.h4_l2_weight}")

    train_loader, _ = get_dataloaders(
        args.data_dir,
        args.target_speaker,
        args.other_speakers,
        batch_size=args.batch_size,
        window_size=args.window_size,
        shift_size=args.shift_size,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers
    )

    converter = MelSpectrogramConverter(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    converter = move_converter_to_device(converter, args.device)

    selector = RuleBasedHeaderSelector(
        sample_rate=args.sample_rate,
        analysis_duration=args.analysis_duration,
        low_energy_threshold=args.low_energy_threshold,
        high_energy_threshold=args.high_energy_threshold,
        high_centroid_threshold=args.high_centroid_threshold,
        high_zcr_threshold=args.high_zcr_threshold,
        noisy_zcr_threshold=args.noisy_zcr_threshold
    )

    training_log = HeaderTrainingLog(args)
    subsets, all_examples, selector_selected_counts = build_condition_subsets(
        train_loader=train_loader,
        selector=selector,
        args=args,
        training_log=training_log
    )

    header_bank = HeaderBank(
        num_headers=args.num_headers,
        mel_bins=args.n_mels,
        time_length=args.header_length,
        device=args.device
    )

    optimizer = optim.Adam(header_bank.parameters(), lr=args.lr)

    speaker_loss_manager = speaker_loss_manager.to(args.device)
    speaker_loss_manager.freeze_all()

    header_update_counts: Dict[int, int] = {idx: 0 for idx in range(args.num_headers)}
    header_weighted_update_counts: Dict[int, float] = {idx: 0.0 for idx in range(args.num_headers)}
    header_loss_sums: Dict[int, float] = {idx: 0.0 for idx in range(args.num_headers)}
    header_loss_counts: Dict[int, int] = {idx: 0 for idx in range(args.num_headers)}

    total_steps = 0

    for epoch in range(args.epochs):
        if total_steps >= args.max_steps:
            break
        epoch_loss = 0.0
        epoch_updates = 0

        print(f"\n[2단계] Epoch {epoch + 1}/{args.epochs}: 조건별 header 학습")

        for current_header_index in range(args.num_headers):
            if total_steps >= args.max_steps:
                break

            training_examples = build_examples_for_header(
                header_index=current_header_index,
                subsets=subsets,
                all_examples=all_examples,
                args=args
            )

            if len(training_examples) == 0:
                print(f"  header {current_header_index}: 학습 가능한 샘플이 없어 건너뜁니다.")
                continue

            print(
                f"  header {current_header_index} 학습 시작 "
                f"(assigned={len(subsets[current_header_index])}, train_examples={len(training_examples)})"
            )

            for example, sample_weight, sample_type in tqdm.tqdm(
                training_examples,
                desc=f"Training header {current_header_index}",
                leave=False
            ):
                if total_steps >= args.max_steps:
                    break

                single_source_waveform = example["source_waveform"].to(args.device)
                single_target_waveform = example["target_waveform"].to(args.device)
                features = example["features"]
                assigned_header = example["assigned_header"]

                source_mel = converter.waveform_to_mel(single_source_waveform).unsqueeze(0)
                target_mel = converter.waveform_to_mel(single_target_waveform).unsqueeze(0)

                source_mel, target_mel = crop_to_header_length(
                    source_mel,
                    target_mel,
                    args.header_length
                )

                if source_mel.shape[-1] == 0 or target_mel.shape[-1] == 0:
                    continue

                weighted_losses = []
                loss_detail_items = []

                # 1. 현재 header의 condition-specific 학습
                header_loss, logs = compute_condition_aware_header_loss(
                    source_mel=source_mel,
                    target_mel=target_mel,
                    header_bank=header_bank,
                    header_index=current_header_index,
                    speaker_loss_manager=speaker_loss_manager,
                    converter=converter,
                    args=args
                )

                training_details = [(current_header_index, sample_weight, logs["total"], sample_type)]
                weighted_losses.append(header_loss * sample_weight)
                loss_detail_items.append(
                    f"h{current_header_index}:{sample_type}:w{sample_weight:.2f}:loss{logs['total']:.4f}"
                )

                header_update_counts[current_header_index] += 1
                header_weighted_update_counts[current_header_index] += sample_weight
                header_loss_sums[current_header_index] += logs["total"]
                header_loss_counts[current_header_index] += 1

                # 2. default header 0은 fallback 안정성을 위해 낮은 가중치로 함께 학습 가능
                #    단, default header 자체를 학습 중일 때는 중복 적용하지 않는다.
                if args.enable_default_update and current_header_index != 0:
                    default_loss, default_logs = compute_condition_aware_header_loss(
                        source_mel=source_mel,
                        target_mel=target_mel,
                        header_bank=header_bank,
                        header_index=0,
                        speaker_loss_manager=speaker_loss_manager,
                        converter=converter,
                        args=args
                    )

                    training_details.append((0, args.default_header_weight, default_logs["total"], "default"))
                    weighted_losses.append(default_loss * args.default_header_weight)
                    loss_detail_items.append(
                        f"h0:default:w{args.default_header_weight:.2f}:loss{default_logs['total']:.4f}"
                    )

                    header_update_counts[0] += 1
                    header_weighted_update_counts[0] += args.default_header_weight
                    header_loss_sums[0] += default_logs["total"]
                    header_loss_counts[0] += 1

                total_weight = sample_weight
                if args.enable_default_update and current_header_index != 0:
                    total_weight += args.default_header_weight

                loss = torch.stack(weighted_losses).sum() / total_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Header Bank 전체 perturbation 크기를 epsilon 범위로 제한한다.
                with torch.no_grad():
                    header_bank.headers.data = torch.clamp(
                        header_bank.headers.data,
                        -args.epsilon,
                        args.epsilon
                    )

                epoch_loss += loss.item()
                epoch_updates += 1
                total_steps += 1
                training_log.training(epoch + 1, total_steps, example["metadata"],
                                      assigned_header, training_details, loss.item())

                if total_steps % args.log_interval == 0:
                    training_log.summary("progress", epoch + 1)
                    print(
                        f"[step {total_steps}] "
                        f"loss={loss.item():.6f}, "
                        f"train_header={current_header_index}, "
                        f"assigned_header={assigned_header}, "
                        f"sample_type={sample_type}, "
                        f"energy={features['energy']:.6f}, "
                        f"zcr={features['zero_crossing_rate']:.6f}, "
                        f"centroid={features['spectral_centroid']:.2f}, "
                        f"speaker={logs.get('speaker', 0.0):.4f}, "
                        f"details={' | '.join(loss_detail_items)}"
                    )

        if epoch_updates == 0:
            raise RuntimeError(
                "이번 epoch에서 Header Bank update가 한 번도 발생하지 않았습니다.\n"
                "원인 후보:\n"
                "1. 데이터셋이 비어 있음\n"
                "2. mel 변환 후 header_length에 해당하는 구간이 없음\n"
                "3. window_size가 너무 작음\n"
                "해결:\n"
                "- smoke test라면 --header_length를 줄이세요.\n"
                "- 논문 기준 실험이라면 --window_size 26400 이상을 사용하세요."
            )

        training_log.summary("epoch-end", epoch + 1)
        avg_loss = epoch_loss / epoch_updates
        print(f"\nEpoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.6f}")

        print("[Selector assigned subset counts]")
        for idx in range(args.num_headers):
            print(f"  header {idx}: {selector_selected_counts[idx]} assigned samples")

        print("[Header update counts]")
        for idx in range(args.num_headers):
            print(f"  header {idx}: {header_update_counts[idx]} updates")

        print("[Header weighted update counts]")
        for idx in range(args.num_headers):
            print(f"  header {idx}: {header_weighted_update_counts[idx]:.2f} weighted updates")

        print("[Header average train loss]")
        for idx in range(args.num_headers):
            if header_loss_counts[idx] == 0:
                print(f"  header {idx}: N/A")
            else:
                print(f"  header {idx}: {header_loss_sums[idx] / header_loss_counts[idx]:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"header_bank_epoch_{epoch + 1}.pt")
            header_bank.save(save_path)
            print(f"Header Bank가 저장되었습니다: {save_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    training_log.summary("complete")
    final_path = os.path.join(args.output_dir, args.output_name)
    header_bank.save(final_path)

    print(f"\n최종 Header Bank가 저장되었습니다: {final_path}")

    print("[최종 Selector assigned subset counts]")
    for idx in range(args.num_headers):
        print(f"  header {idx}: {selector_selected_counts[idx]} assigned samples")

    print("[최종 Header update counts]")
    for idx in range(args.num_headers):
        print(f"  header {idx}: {header_update_counts[idx]} updates")

    print("[최종 Header weighted update counts]")
    for idx in range(args.num_headers):
        print(f"  header {idx}: {header_weighted_update_counts[idx]:.2f} weighted updates")

    print("[최종 Header average train loss]")
    for idx in range(args.num_headers):
        if header_loss_counts[idx] == 0:
            print(f"  header {idx}: N/A")
        else:
            print(f"  header {idx}: {header_loss_sums[idx] / header_loss_counts[idx]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="조건별 subset 기반 Header Bank 학습")

    # 데이터셋 인자
    parser.add_argument("--data_dir", type=str, default="./data/VCTK-Corpus",
                        help="데이터셋 루트 디렉터리")
    parser.add_argument("--target_speaker", type=str, required=True,
                        help="목표 화자 ID. 보호 대상")
    parser.add_argument("--other_speakers", type=str, nargs="+", required=True,
                        help="기타 화자 ID 목록. 오도 목적으로 사용")

    # 오디오 인자
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="오디오 샘플링 레이트")
    parser.add_argument("--n_fft", type=int, default=1024,
                        help="FFT 크기")
    parser.add_argument("--hop_length", type=int, default=256,
                        help="프레임 이동 간격")
    parser.add_argument("--n_mels", type=int, default=80,
                        help="멜 필터뱅크 개수")

    # Header Bank 인자
    parser.add_argument("--num_headers", type=int, default=5,
                        help="Header Bank에 포함할 header 개수")
    parser.add_argument("--header_length", type=int, default=100,
                        help="header 시간 길이. mel frame 단위")

    # AntiFake-style encoder option
    parser.add_argument("--external_root", type=str, default="./external",
                        help="외부 encoder/model 루트. 기본 구조: external/rtvc, external/saved_models")
    parser.add_argument("--dummy_loss", type=str2bool, default=DUMMY_LOSS_DEFAULT,
                        help="기존 dummy encoder loss 사용 여부")
    parser.add_argument("--rtvc_loss", type=str2bool, default=RTVC_LOSS_DEFAULT,
                        help="RTVC/SV2TTS speaker encoder loss 사용 여부")
    parser.add_argument("--avc_loss", type=str2bool, default=AVC_LOSS_DEFAULT,
                        help="AVC encoder loss 사용 여부. 현재 adapter 미구현")
    parser.add_argument("--coqui_loss", type=str2bool, default=COQUI_LOSS_DEFAULT,
                        help="Coqui/YourTTS encoder loss 사용 여부. 현재 adapter 미구현")
    parser.add_argument("--tortoise_autoregressive_loss", type=str2bool,
                        default=TORTOISE_AUTOREGRESSIVE_LOSS_DEFAULT,
                        help="Tortoise autoregressive encoder loss 사용 여부. 현재 adapter 미구현")
    parser.add_argument("--tortoise_diffusion_loss", type=str2bool,
                        default=TORTOISE_DIFFUSION_LOSS_DEFAULT,
                        help="Tortoise diffusion encoder loss 사용 여부. 현재 adapter 미구현")
    parser.add_argument("--rtvc_threshold", type=float, default=1.0,
                        help="RTVC threshold-base loss의 거리 threshold. smoke test 후 조정 필요")
    parser.add_argument("--rtvc_scale", type=float, default=1.0,
                        help="RTVC speaker loss scale")
    parser.add_argument("--dummy_scale", type=float, default=1.0,
                        help="Dummy speaker loss scale")

    # 학습 인자
    parser.add_argument("--batch_size", type=int, default=1,
                        help="배치 크기")
    parser.add_argument("--window_size", type=int, default=26400,
                        help="슬라이딩 윈도우 크기. waveform sample 단위. 기본값 26400은 16kHz 기준 1.65초")
    parser.add_argument("--shift_size", type=int, default=6400,
                        help="슬라이딩 윈도우 이동 간격. waveform sample 단위. 기본값 6400은 16kHz 기준 0.4초")
    parser.add_argument("--epochs", type=int, default=10,
                        help="학습 epoch 수")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="전체 학습 step 수 상한. smoke test에서는 작게 설정")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="학습률")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="교란의 최대 크기")
    parser.add_argument("--lambda_param", type=float, default=1.0,
                        help="speaker loss에서 source embedding에서 멀어지는 항의 가중치")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Header Bank 저장 epoch 간격")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="학습 로그 출력 step 간격")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="데이터 로딩 worker 수. WSL/CPU 디버깅에서는 0 권장")
    parser.add_argument("--max_subset_samples", type=int, default=0,
                        help="subset 구성에 사용할 최대 샘플 수. 0이면 전체 사용")

    # subset 기반 hybrid 학습 인자
    parser.add_argument("--min_subset_size", type=int, default=10,
                        help="각 header가 최소한 확보해야 하는 subset 크기. 부족하면 borrowed sample을 사용")
    parser.add_argument("--primary_sample_weight", type=float, default=1.0,
                        help="해당 header에 원래 배정된 primary sample의 학습 가중치")
    parser.add_argument("--enable_default_update", action="store_true",
                        help="non-default header 학습 시 fallback header 0도 함께 낮은 가중치로 업데이트")
    parser.add_argument("--default_header_weight", type=float, default=0.2,
                        help="fallback header 0을 함께 학습할 때의 가중치")
    parser.add_argument("--enable_borrowed_samples", action="store_true",
                        help="subset이 부족한 header에 다른 subset sample을 낮은 가중치로 보충")
    parser.add_argument("--borrowed_sample_weight", type=float, default=0.1,
                        help="부족한 subset 보충용 borrowed sample의 가중치. default_header_weight보다 클 수 없음")

    # selector 인자
    parser.add_argument("--analysis_duration", type=float, default=0.3,
                        help="selector가 분석할 발화 시작 구간 길이. 초 단위")
    parser.add_argument("--low_energy_threshold", type=float, default=0.01,
                        help="저에너지 시작 구간 판정 threshold")
    parser.add_argument("--high_energy_threshold", type=float, default=0.08,
                        help="고에너지 시작 구간 판정 threshold")
    parser.add_argument("--high_centroid_threshold", type=float, default=3000.0,
                        help="고주파성 시작 구간 판정 spectral centroid threshold")
    parser.add_argument("--high_zcr_threshold", type=float, default=0.18,
                        help="무성음/마찰음 계열 판정 zero-crossing rate threshold")
    parser.add_argument("--noisy_zcr_threshold", type=float, default=0.25,
                        help="잡음 가능 구간 판정 zero-crossing rate threshold")

    # header별 추가 loss/제약 인자
    parser.add_argument("--h0_l2_weight", type=float, default=0.05,
                        help="Header 0 fallback 안정성을 위한 L2 regularization 가중치")
    parser.add_argument("--h0_smooth_weight", type=float, default=0.02,
                        help="Header 0 fallback 안정성을 위한 temporal smoothness 가중치")

    parser.add_argument("--h1_norm_weight", type=float, default=0.15,
                        help="Header 1 저에너지 구간용 강한 perturbation norm penalty")
    parser.add_argument("--h1_smooth_weight", type=float, default=0.05,
                        help="Header 1 저에너지 구간용 smoothness penalty")

    parser.add_argument("--h2_attack_boost", type=float, default=1.3,
                        help="Header 2 고에너지/갑작스러운 시작 구간의 speaker attack loss 증폭 계수")
    parser.add_argument("--h2_onset_attack_weight", type=float, default=0.5,
                        help="Header 2 onset region speaker loss 추가 가중치")
    parser.add_argument("--h2_onset_frames", type=int, default=20,
                        help="Header 2 onset attack loss를 계산할 앞쪽 mel frame 수")
    parser.add_argument("--h2_l2_weight", type=float, default=0.03,
                        help="Header 2의 과도한 perturbation을 막기 위한 L2 penalty")

    parser.add_argument("--h3_high_freq_weight", type=float, default=0.10,
                        help="Header 3 고주파 mel bin perturbation penalty")
    parser.add_argument("--h3_high_freq_start_ratio", type=float, default=0.65,
                        help="Header 3에서 고주파 영역으로 볼 mel bin 시작 비율")
    parser.add_argument("--h3_l2_weight", type=float, default=0.03,
                        help="Header 3의 전체 L2 penalty")

    parser.add_argument("--h4_consistency_weight", type=float, default=0.10,
                        help="Header 4 noise-robust consistency loss 가중치")
    parser.add_argument("--h4_noise_std", type=float, default=0.01,
                        help="Header 4 consistency loss에 사용할 mel-domain noise 표준편차")
    parser.add_argument("--h4_l2_weight", type=float, default=0.03,
                        help="Header 4의 전체 L2 penalty")

    parser.add_argument("--extra_header_l2_weight", type=float, default=0.03,
                        help="num_headers가 5보다 클 경우 추가 header에 적용할 기본 L2 penalty")

    # 기타 인자
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="사용할 장치")
    parser.add_argument("--output_dir", type=str, default="./output/header_bank",
                        help="출력 디렉터리")
    parser.add_argument("--output_name", type=str, default="header_bank.pt",
                        help="저장할 Header Bank 파일명")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. device를 cpu로 변경합니다.")
        args.device = "cpu"

    if args.num_headers < 5:
        raise ValueError(
            "현재 header별 loss 설계는 header 0~4를 전제로 합니다. --num_headers는 최소 5 이상이어야 합니다."
        )

    if args.borrowed_sample_weight > args.default_header_weight:
        raise ValueError(
            "borrowed_sample_weight는 default_header_weight보다 클 수 없습니다.\n"
            f"현재 borrowed_sample_weight={args.borrowed_sample_weight}, "
            f"default_header_weight={args.default_header_weight}\n"
            "다른 header에 대한 보조 학습 가중치는 default header보다 낮거나 같아야 합니다."
        )

    if args.h3_high_freq_start_ratio < 0.0 or args.h3_high_freq_start_ratio >= 1.0:
        raise ValueError("--h3_high_freq_start_ratio는 0.0 이상 1.0 미만이어야 합니다.")

    # 기본적으로 논문 설계에 맞게 fallback/default update와 borrowed sample 보충을 켠다.
    # 사용자가 명령어에서 명시적으로 끄고 싶다면 argparse 구조를 별도로 바꿔야 하므로,
    # 현재 연구용 기본값은 True로 보정한다.
    if not args.enable_default_update:
        args.enable_default_update = True

    if not args.enable_borrowed_samples:
        args.enable_borrowed_samples = True

    speaker_loss_manager = SpeakerLossManager(args).to(args.device)

    train_header_bank(speaker_loss_manager, args)


if __name__ == "__main__":
    main()
