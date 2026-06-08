import os
import argparse
from typing import Dict, Tuple, List, Any
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from data.dataset import get_dataloaders
from models.header_bank import HeaderBank
from utils.audio import MelSpectrogramConverter
from utils.header_selector import RuleBasedHeaderSelector


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


def compute_speaker_embedding_loss(source_mel: torch.Tensor,
                                   target_mel: torch.Tensor,
                                   perturbed_mel: torch.Tensor,
                                   speaker_encoder: nn.Module,
                                   lambda_param: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Header Bank의 기본 speaker embedding loss를 계산한다.

    목표:
        1. perturbed embedding이 target speaker embedding에 가까워지게 함
        2. perturbed embedding이 source speaker embedding에서 멀어지게 함

    Args:
        source_mel: 원본 화자 mel [1, 1, F, T]
        target_mel: 목표 화자 mel [1, 1, F, T]
        perturbed_mel: header가 적용된 mel [1, 1, F, T]
        speaker_encoder: 화자 인코더
        lambda_param: source embedding에서 멀어지는 항의 가중치

    Returns:
        speaker_loss, 로그용 loss dict
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
                               speaker_encoder: nn.Module,
                               lambda_param: float,
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

    onset_loss, _ = compute_speaker_embedding_loss(
        source_mel=onset_source,
        target_mel=onset_target,
        perturbed_mel=onset_perturbed,
        speaker_encoder=speaker_encoder,
        lambda_param=lambda_param
    )

    return onset_loss


def compute_noise_consistency_loss(source_mel: torch.Tensor,
                                   perturbed_mel: torch.Tensor,
                                   header_bank: HeaderBank,
                                   header_index: int,
                                   speaker_encoder: nn.Module,
                                   noise_std: float) -> torch.Tensor:
    """
    Header 4의 noise-robust consistency loss.
    source_mel에 작은 잡음을 더한 뒤 같은 header를 적용했을 때,
    깨끗한 입력에서의 perturbed embedding과 비슷한 방향으로 유지되도록 한다.
    """
    if noise_std <= 0:
        return source_mel.new_tensor(0.0)

    noisy_source = source_mel + torch.randn_like(source_mel) * noise_std
    noisy_perturbed = header_bank.apply_header(noisy_source, header_index)

    clean_embedding = speaker_encoder(perturbed_mel)
    noisy_embedding = speaker_encoder(noisy_perturbed)

    return F.mse_loss(noisy_embedding, clean_embedding.detach())


def compute_condition_aware_header_loss(source_mel: torch.Tensor,
                                        target_mel: torch.Tensor,
                                        header_bank: HeaderBank,
                                        header_index: int,
                                        speaker_encoder: nn.Module,
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

    speaker_loss, logs = compute_speaker_embedding_loss(
        source_mel=source_mel,
        target_mel=target_mel,
        perturbed_mel=perturbed_mel,
        speaker_encoder=speaker_encoder,
        lambda_param=args.lambda_param
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
            speaker_encoder=speaker_encoder,
            lambda_param=args.lambda_param,
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
            speaker_encoder=speaker_encoder,
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
                            args) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[int, int]]:
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
            header_index = normalize_header_index(header_index, args.num_headers)

            normalized_features = {
                "energy": safe_float(features.get("energy", 0.0)),
                "zero_crossing_rate": safe_float(features.get("zero_crossing_rate", 0.0)),
                "spectral_centroid": safe_float(features.get("spectral_centroid", 0.0))
            }

            example = {
                "source_waveform": single_source_waveform,
                "target_waveform": single_target_waveform,
                "assigned_header": header_index,
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


def train_header_bank(speaker_encoder: nn.Module, args) -> None:
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

    subsets, all_examples, selector_selected_counts = build_condition_subsets(
        train_loader=train_loader,
        selector=selector,
        args=args
    )

    header_bank = HeaderBank(
        num_headers=args.num_headers,
        mel_bins=args.n_mels,
        time_length=args.header_length,
        device=args.device
    )

    optimizer = optim.Adam(header_bank.parameters(), lr=args.lr)

    speaker_encoder = speaker_encoder.to(args.device)
    freeze_model(speaker_encoder)

    header_update_counts: Dict[int, int] = {idx: 0 for idx in range(args.num_headers)}
    header_weighted_update_counts: Dict[int, float] = {idx: 0.0 for idx in range(args.num_headers)}
    header_loss_sums: Dict[int, float] = {idx: 0.0 for idx in range(args.num_headers)}
    header_loss_counts: Dict[int, int] = {idx: 0 for idx in range(args.num_headers)}

    total_steps = 0

    for epoch in range(args.epochs):
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
                    speaker_encoder=speaker_encoder,
                    args=args
                )

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
                        speaker_encoder=speaker_encoder,
                        args=args
                    )

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

                if total_steps % args.log_interval == 0:
                    print(
                        f"[step {total_steps}] "
                        f"loss={loss.item():.6f}, "
                        f"train_header={current_header_index}, "
                        f"assigned_header={assigned_header}, "
                        f"sample_type={sample_type}, "
                        f"energy={features['energy']:.6f}, "
                        f"zcr={features['zero_crossing_rate']:.6f}, "
                        f"centroid={features['spectral_centroid']:.2f}, "
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

    # TODO: 여기서는 목표 음성 합성 모델의 화자 인코더를 로드해야 함
    # 아래 코드는 예시이며, 실제 적용 시에는 실제 인코더를 로드해야 함
    class DummySpeakerEncoder(nn.Module):
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

    speaker_encoder = DummySpeakerEncoder().to(args.device)

    train_header_bank(speaker_encoder, args)


if __name__ == "__main__":
    main()
