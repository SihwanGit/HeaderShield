import os
import argparse
from typing import Dict, Tuple, List

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


def move_converter_to_device(converter: MelSpectrogramConverter, device: str) -> MelSpectrogramConverter:
    """
    MelSpectrogramConverter 내부 torchaudio transform들을 현재 device로 이동한다.
    GPU 환경에서 waveform은 cuda에 있는데 mel_transform은 cpu에 남아 있으면 device mismatch가 발생할 수 있다.
    """
    # 수정: utils/audio.py에 to() 메서드를 추가한 경우 해당 메서드를 우선 사용
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


def compute_header_bank_loss(source_mel: torch.Tensor,
                             target_mel: torch.Tensor,
                             perturbed_mel: torch.Tensor,
                             speaker_encoder: nn.Module,
                             lambda_param: float) -> torch.Tensor:
    """
    Header Bank 학습용 speaker embedding loss를 계산한다.

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
        최종 loss
    """
    # 수정: source/target embedding은 header에 의해 바뀌지 않으므로 no_grad로 계산
    with torch.no_grad():
        source_embedding = speaker_encoder(source_mel).detach()
        target_embedding = speaker_encoder(target_mel).detach()

    perturbed_embedding = speaker_encoder(perturbed_mel)

    loss_target = F.mse_loss(perturbed_embedding, target_embedding)
    loss_source = F.mse_loss(perturbed_embedding, source_embedding)

    loss = loss_target - lambda_param * loss_source

    return loss


def add_header_weight(plan: Dict[int, float],
                      header_index: int,
                      weight: float,
                      num_headers: int) -> None:
    """
    학습 계획(plan)에 header index와 loss weight를 추가한다.
    이미 같은 header가 들어있으면 weight를 누적한다.

    Args:
        plan: {header_index: weight}
        header_index: 학습할 header 번호
        weight: 해당 header loss 가중치
        num_headers: 전체 header 개수
    """
    if weight <= 0:
        return

    if header_index < 0:
        header_index = 0

    if header_index >= num_headers:
        header_index = num_headers - 1

    if header_index not in plan:
        plan[header_index] = 0.0

    plan[header_index] += weight


def build_header_training_plan(selected_header_index: int,
                               total_steps: int,
                               args) -> List[Tuple[int, float]]:
    """
    현재 step에서 어떤 header들을 학습할지 결정한다.

    training_strategy:
        selected_only:
            selector가 고른 header만 학습한다.
            기존 train_header_bank.py와 같은 방식이다.

        cycle_only:
            selector 결과와 무관하게 header를 0,1,2,3,4 순서로 강제 순환 학습한다.
            데이터가 부족해도 모든 header가 업데이트되도록 하는 방식이다.

        hybrid:
            selector가 고른 header를 가장 강하게 학습하고,
            default header 0과 cycle header를 보조적으로 함께 학습한다.
            방법2 + 방법3을 결합한 안정화 방식이다.

    Returns:
        [(header_index, weight), ...]
    """
    plan: Dict[int, float] = {}

    cycle_header_index = total_steps % args.num_headers

    if args.training_strategy == "selected_only":
        add_header_weight(
            plan=plan,
            header_index=selected_header_index,
            weight=1.0,
            num_headers=args.num_headers
        )

    elif args.training_strategy == "cycle_only":
        add_header_weight(
            plan=plan,
            header_index=cycle_header_index,
            weight=1.0,
            num_headers=args.num_headers
        )

    elif args.training_strategy == "hybrid":
        # 1. selector가 고른 header를 가장 강하게 학습한다.
        add_header_weight(
            plan=plan,
            header_index=selected_header_index,
            weight=args.selected_header_weight,
            num_headers=args.num_headers
        )

        # 2. default header 0을 fallback header로 함께 학습한다.
        if args.enable_default_update:
            add_header_weight(
                plan=plan,
                header_index=0,
                weight=args.default_header_weight,
                num_headers=args.num_headers
            )

        # 3. 모든 header가 최소한 주기적으로 학습되도록 cycle header를 함께 학습한다.
        if args.enable_cycle_update:
            add_header_weight(
                plan=plan,
                header_index=cycle_header_index,
                weight=args.cycle_header_weight,
                num_headers=args.num_headers
            )

    else:
        raise ValueError(f"지원하지 않는 training_strategy입니다: {args.training_strategy}")

    return sorted(plan.items(), key=lambda x: x[0])


def train_header_bank(speaker_encoder: nn.Module, args) -> None:
    """
    Selector 기반 Header Bank 학습.

    기존 방식:
        1. source waveform의 발화 시작 구간을 selector로 분석
        2. selector가 header index 선택
        3. 선택된 header만 source mel 앞부분에 적용
        4. 선택된 header만 업데이트

    수정된 방식:
        selected_only:
            기존 방식과 동일하게 selector가 고른 header만 학습한다.

        cycle_only:
            모든 header가 최소한 한 번씩 학습되도록 header index를 순환시킨다.

        hybrid:
            selector가 고른 header를 중심으로 학습하되,
            default header 0과 cycle header를 함께 업데이트하여
            데이터 부족 상황에서도 특정 header가 미학습 상태로 남는 문제를 완화한다.

    Header index 의미:
        0: default
        1: low-energy start
        2: high-energy / abrupt start
        3: high spectral centroid / fricative-like start
        4: noisy-like start
    """
    print("Selector 기반 Header Bank 학습을 시작합니다...")

    print("[설정 확인]")
    print(f"  data_dir              : {args.data_dir}")
    print(f"  target_speaker        : {args.target_speaker}")
    print(f"  other_speakers        : {args.other_speakers}")
    print(f"  num_headers           : {args.num_headers}")
    print(f"  sample_rate           : {args.sample_rate}")
    print(f"  window_size           : {args.window_size} samples ({args.window_size / args.sample_rate:.3f} sec)")
    print(f"  shift_size            : {args.shift_size} samples ({args.shift_size / args.sample_rate:.3f} sec)")
    print(f"  hop_length            : {args.hop_length} samples")
    print(f"  header_length         : {args.header_length} mel frames ({args.header_length * args.hop_length / args.sample_rate:.3f} sec)")
    print(f"  device                : {args.device}")
    print(f"  training_strategy     : {args.training_strategy}")
    print(f"  selected_header_weight: {args.selected_header_weight}")
    print(f"  enable_default_update : {args.enable_default_update}")
    print(f"  default_header_weight : {args.default_header_weight}")
    print(f"  enable_cycle_update   : {args.enable_cycle_update}")
    print(f"  cycle_header_weight   : {args.cycle_header_weight}")

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

    header_bank = HeaderBank(
        num_headers=args.num_headers,
        mel_bins=args.n_mels,
        time_length=args.header_length,
        device=args.device
    )

    optimizer = optim.Adam(header_bank.parameters(), lr=args.lr)

    speaker_encoder = speaker_encoder.to(args.device)
    freeze_model(speaker_encoder)

    # selector가 실제로 어떤 header를 선택했는지 기록
    selector_selected_counts: Dict[int, int] = {idx: 0 for idx in range(args.num_headers)}

    # optimizer step에서 실제로 어떤 header가 학습되었는지 기록
    header_update_counts: Dict[int, int] = {idx: 0 for idx in range(args.num_headers)}

    # loss weight까지 반영한 누적 학습량 기록
    header_weighted_update_counts: Dict[int, float] = {idx: 0.0 for idx in range(args.num_headers)}

    training_log = HeaderTrainingLog(args)
    total_steps = 0

    for epoch in range(args.epochs):
        if total_steps >= args.max_steps:
            break
        epoch_loss = 0.0
        epoch_updates = 0

        print(f"Epoch {epoch + 1}/{args.epochs}")

        for batch in tqdm.tqdm(train_loader):
            source_waveform = batch["source_waveform"].to(args.device)
            target_waveform = batch["target_waveform"].to(args.device)

            for i in range(source_waveform.size(0)):
                if total_steps >= args.max_steps:
                    break

                single_source_waveform = source_waveform[i]
                single_target_waveform = target_waveform[i]

                # selector는 waveform 시작 구간을 보고 header index를 결정한다.
                selected_header_index, features = selector.select(
                    single_source_waveform.detach().cpu(),
                    return_features=True
                )

                raw_header_index = selected_header_index
                metadata = slice_metadata(batch, i, args.sample_rate)

                # HeaderBank의 num_headers보다 큰 index가 선택될 가능성을 방어
                if selected_header_index >= args.num_headers:
                    selected_header_index = args.num_headers - 1

                if selected_header_index < 0:
                    selected_header_index = 0

                selector_selected_counts[selected_header_index] += 1
                training_log.assignment(metadata, raw_header_index, selected_header_index, features)

                source_mel = converter.waveform_to_mel(single_source_waveform).unsqueeze(0)
                target_mel = converter.waveform_to_mel(single_target_waveform).unsqueeze(0)

                source_mel, target_mel = crop_to_header_length(
                    source_mel,
                    target_mel,
                    args.header_length
                )

                if source_mel.shape[-1] == 0 or target_mel.shape[-1] == 0:
                    continue

                # 현재 step에서 학습할 header 목록을 만든다.
                # hybrid 전략에서는 selected header + default header + cycle header가 함께 들어갈 수 있다.
                training_plan = build_header_training_plan(
                    selected_header_index=selected_header_index,
                    total_steps=total_steps,
                    args=args
                )

                if len(training_plan) == 0:
                    raise RuntimeError("학습할 header가 없습니다. training_strategy 설정을 확인하세요.")

                training_details = []
                weighted_losses = []
                loss_detail_items = []

                for train_header_index, weight in training_plan:
                    # HeaderBank.apply_header가 int index를 받는 구조를 유지한다.
                    perturbed_mel = header_bank.apply_header(source_mel, train_header_index)

                    header_loss = compute_header_bank_loss(
                        source_mel=source_mel,
                        target_mel=target_mel,
                        perturbed_mel=perturbed_mel,
                        speaker_encoder=speaker_encoder,
                        lambda_param=args.lambda_param
                    )

                    training_details.append((train_header_index, weight, header_loss.item(), args.training_strategy))
                    weighted_losses.append(header_loss * weight)
                    loss_detail_items.append(f"h{train_header_index}:w{weight:.2f}:loss{header_loss.item():.4f}")

                    header_update_counts[train_header_index] += 1
                    header_weighted_update_counts[train_header_index] += weight

                total_weight = sum(weight for _, weight in training_plan)

                # step마다 학습되는 header 개수나 weight 합이 달라질 수 있으므로 weight 합으로 나눠 loss scale을 안정화한다.
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
                training_log.training(epoch + 1, total_steps, metadata,
                                      selected_header_index, training_details, loss.item())

                if total_steps % args.log_interval == 0:
                    training_log.summary("progress", epoch + 1)
                    energy = features.get("energy", 0.0)
                    zcr = features.get("zero_crossing_rate", 0.0)
                    centroid = features.get("spectral_centroid", 0.0)

                    print(
                        f"[step {total_steps}] "
                        f"loss={loss.item():.6f}, "
                        f"selected_header={selected_header_index}, "
                        f"train_plan={training_plan}, "
                        f"energy={energy:.6f}, "
                        f"zcr={zcr:.6f}, "
                        f"centroid={centroid:.2f}, "
                        f"details={' | '.join(loss_detail_items)}"
                    )

            if total_steps >= args.max_steps:
                break

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
        print(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.6f}")

        print("[Selector selected counts]")
        for idx in range(args.num_headers):
            print(f"  header {idx}: {selector_selected_counts[idx]} selected")

        print("[Header update counts]")
        for idx in range(args.num_headers):
            print(f"  header {idx}: {header_update_counts[idx]} updates")

        print("[Header weighted update counts]")
        for idx in range(args.num_headers):
            print(f"  header {idx}: {header_weighted_update_counts[idx]:.2f} weighted updates")

        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"header_bank_epoch_{epoch + 1}.pt")
            header_bank.save(save_path)
            print(f"Header Bank가 저장되었습니다: {save_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    training_log.summary("complete")
    final_path = os.path.join(args.output_dir, args.output_name)
    header_bank.save(final_path)

    print(f"최종 Header Bank가 저장되었습니다: {final_path}")

    print("[최종 Selector selected counts]")
    for idx in range(args.num_headers):
        print(f"  header {idx}: {selector_selected_counts[idx]} selected")

    print("[최종 Header update counts]")
    for idx in range(args.num_headers):
        print(f"  header {idx}: {header_update_counts[idx]} updates")

    print("[최종 Header weighted update counts]")
    for idx in range(args.num_headers):
        print(f"  header {idx}: {header_weighted_update_counts[idx]:.2f} weighted updates")


def main():
    parser = argparse.ArgumentParser(description="Selector 기반 Header Bank 학습")

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
                        help="손실 함수에서 사용하는 균형 파라미터")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Header Bank 저장 epoch 간격")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="학습 로그 출력 step 간격")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="데이터 로딩 worker 수. WSL/CPU 디버깅에서는 0 권장")

    # Header Bank 학습 전략 인자
    parser.add_argument("--training_strategy", type=str, default="hybrid",
                        choices=["selected_only", "cycle_only", "hybrid"],
                        help="Header Bank 학습 전략. selected_only는 기존 방식, cycle_only는 순환 학습, hybrid는 선택 header+default+cycle 결합")
    parser.add_argument("--selected_header_weight", type=float, default=1.0,
                        help="hybrid 전략에서 selector가 고른 header의 loss 가중치")
    parser.add_argument("--enable_default_update", action="store_true",
                        help="hybrid 전략에서 default header 0을 함께 업데이트할지 여부")
    parser.add_argument("--default_header_weight", type=float, default=0.3,
                        help="default header 0을 함께 학습할 때의 loss 가중치")
    parser.add_argument("--enable_cycle_update", action="store_true",
                        help="hybrid 전략에서 cycle header를 함께 업데이트할지 여부")
    parser.add_argument("--cycle_header_weight", type=float, default=0.5,
                        help="cycle header를 함께 학습할 때의 loss 가중치")

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

    # hybrid 전략에서는 기본적으로 default update와 cycle update를 켜는 것이 목적에 맞다.
    # argparse의 action="store_true"는 명령어에 옵션을 넣어야 True가 되므로,
    # 사용자가 selected_only나 cycle_only를 쓰지 않는 경우에는 안전하게 True로 보정한다.
    if args.training_strategy == "hybrid":
        if not args.enable_default_update:
            args.enable_default_update = True
        if not args.enable_cycle_update:
            args.enable_cycle_update = True

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
