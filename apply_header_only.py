import os
import argparse
from typing import Tuple, Optional

import torch
import torchaudio

from models.header_model import UniversalPerturbationHeader
from models.header_bank import HeaderBank
from utils.audio import MelSpectrogramConverter, match_waveform_length
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


def load_waveform(input_path: str, sample_rate: int, device: str) -> Tuple[torch.Tensor, int]:
    """
    wav 파일을 로드하고 header 적용에 맞게 전처리한다.

    Args:
        input_path: 입력 wav 파일 경로
        sample_rate: 목표 sample rate
        device: 사용할 device

    Returns:
        waveform: mono waveform [1, T]
        sample_rate: 저장에 사용할 sample rate
    """
    waveform, sr = torchaudio.load(input_path)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = torch.clamp(waveform, -1.0, 1.0)
    waveform = waveform.to(device)

    return waveform, sr


def apply_fixed_header(
    mel_spec: torch.Tensor,
    header_path: str,
    device: str,
    n_mels: int,
    header_length: int,
) -> torch.Tensor:
    """
    fixed universal header만 적용한다.

    Args:
        mel_spec: 입력 mel-spectrogram [B, 1, F, T]
        header_path: universal_header.pt 경로
        device: 사용할 device
        n_mels: mel bin 개수
        header_length: header 길이. mel frame 단위

    Returns:
        fixed header가 적용된 mel-spectrogram [B, 1, F, T]
    """
    header = UniversalPerturbationHeader(
        mel_bins=n_mels,
        time_length=header_length,
        device=device,
    )
    header.load(header_path)

    perturbed_mel = header.apply_header(mel_spec)

    return perturbed_mel


def apply_header_bank(
    mel_spec: torch.Tensor,
    waveform: torch.Tensor,
    header_bank_path: str,
    device: str,
    n_mels: int,
    num_headers: int,
    header_length: int,
    selector: RuleBasedHeaderSelector,
) -> Tuple[torch.Tensor, int, dict]:
    """
    Header Bank + Selector 방식으로 선택된 header만 적용한다.

    Args:
        mel_spec: 입력 mel-spectrogram [B, 1, F, T]
        waveform: 입력 waveform [1, T]
        header_bank_path: header_bank.pt 경로
        device: 사용할 device
        n_mels: mel bin 개수
        num_headers: Header Bank에 포함된 header 개수
        header_length: header 길이. mel frame 단위
        selector: RuleBasedHeaderSelector

    Returns:
        perturbed_mel: 선택된 header가 적용된 mel-spectrogram
        header_index: 선택된 header index
        features: selector가 계산한 시작 구간 feature
    """
    header_bank = HeaderBank(
        num_headers=num_headers,
        mel_bins=n_mels,
        time_length=header_length,
        device=device,
    )
    header_bank.load(header_bank_path)

    header_index, features = selector.select(
        waveform.detach().cpu(),
        return_features=True,
    )

    if header_index >= header_bank.num_headers:
        header_index = header_bank.num_headers - 1

    perturbed_mel = header_bank.apply_header(mel_spec, header_index)

    return perturbed_mel, header_index, features


def protect_header_only(args) -> None:
    """
    predictive model 없이 header만 적용하여 보호 음성을 생성한다.

    비교 목적:
        1. Raw
        2. Fixed Universal Header only
        3. Header Bank + Selector only

    이 코드는 위 2, 3번을 만들기 위한 header-only 평가용 스크립트이다.
    """
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. device를 cpu로 변경합니다.")
        args.device = "cpu"

    print("[Header-only 보호 시작]")
    print(f"  mode          : {args.mode}")
    print(f"  input         : {args.input}")
    print(f"  output        : {args.output}")
    print(f"  sample_rate   : {args.sample_rate}")
    print(f"  n_fft         : {args.n_fft}")
    print(f"  hop_length    : {args.hop_length}")
    print(f"  n_mels        : {args.n_mels}")
    print(f"  header_length : {args.header_length}")
    print(f"  device        : {args.device}")

    waveform, sr = load_waveform(
        input_path=args.input,
        sample_rate=args.sample_rate,
        device=args.device,
    )
    original_length = waveform.shape[-1]

    converter = MelSpectrogramConverter(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    converter = move_converter_to_device(converter, args.device)

    mel_spec = converter.waveform_to_mel(waveform)

    # waveform_to_mel은 [1, F, T]를 반환하므로 header 입력에 맞게 [B, 1, F, T]로 변환
    if mel_spec.dim() == 3:
        mel_spec = mel_spec.unsqueeze(0)

    selected_header_index: Optional[int] = None
    selector_features: Optional[dict] = None

    if args.mode == "fixed":
        if not args.header:
            raise ValueError("--mode fixed에서는 --header 경로가 필요합니다.")

        perturbed_mel = apply_fixed_header(
            mel_spec=mel_spec,
            header_path=args.header,
            device=args.device,
            n_mels=args.n_mels,
            header_length=args.header_length,
        )

    elif args.mode == "bank":
        if not args.header_bank:
            raise ValueError("--mode bank에서는 --header_bank 경로가 필요합니다.")

        selector = RuleBasedHeaderSelector(
            sample_rate=args.sample_rate,
            analysis_duration=args.analysis_duration,
            low_energy_threshold=args.low_energy_threshold,
            high_energy_threshold=args.high_energy_threshold,
            high_centroid_threshold=args.high_centroid_threshold,
            high_zcr_threshold=args.high_zcr_threshold,
            noisy_zcr_threshold=args.noisy_zcr_threshold,
        )

        perturbed_mel, selected_header_index, selector_features = apply_header_bank(
            mel_spec=mel_spec,
            waveform=waveform,
            header_bank_path=args.header_bank,
            device=args.device,
            n_mels=args.n_mels,
            num_headers=args.num_headers,
            header_length=args.header_length,
            selector=selector,
        )

    else:
        raise ValueError(f"지원하지 않는 mode입니다: {args.mode}")

    protected_waveform = converter.mel_to_waveform(perturbed_mel)
    protected_waveform = match_waveform_length(protected_waveform, original_length)
    protected_waveform = torch.clamp(protected_waveform, -1.0, 1.0)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torchaudio.save(args.output, protected_waveform.cpu(), sr)

    print("[Header-only 보호 완료]")
    print(f"  saved_to: {args.output}")

    if selected_header_index is not None:
        print("[Selector 결과]")
        print(f"  selected_header_index: {selected_header_index}")
        if selector_features is not None:
            print(f"  energy               : {selector_features['energy']:.6f}")
            print(f"  zero_crossing_rate   : {selector_features['zero_crossing_rate']:.6f}")
            print(f"  spectral_centroid    : {selector_features['spectral_centroid']:.2f}")
            print(f"  analysis_duration    : {selector_features['analysis_duration']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Header-only protection script for Fixed Header vs Header Bank comparison")

    # 실행 모드
    parser.add_argument("--mode", type=str, required=True, choices=["fixed", "bank"],
                        help="fixed: universal_header.pt 적용, bank: header_bank.pt + selector 적용")

    # 입력/출력
    parser.add_argument("--input", type=str, required=True,
                        help="입력 wav 파일 경로")
    parser.add_argument("--output", type=str, required=True,
                        help="출력 wav 파일 경로")

    # fixed header / header bank 경로
    parser.add_argument("--header", type=str, default=None,
                        help="fixed universal header 경로. --mode fixed에서 필요")
    parser.add_argument("--header_bank", type=str, default=None,
                        help="Header Bank 경로. --mode bank에서 필요")

    # 오디오 인자
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="오디오 샘플링 레이트")
    parser.add_argument("--n_fft", type=int, default=1024,
                        help="FFT 크기")
    parser.add_argument("--hop_length", type=int, default=256,
                        help="프레임 이동 간격")
    parser.add_argument("--n_mels", type=int, default=80,
                        help="멜 필터뱅크 개수")

    # header 인자
    parser.add_argument("--num_headers", type=int, default=5,
                        help="Header Bank에 포함된 header 개수")
    parser.add_argument("--header_length", type=int, default=100,
                        help="header 길이. mel frame 단위")

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

    args = parser.parse_args()

    protect_header_only(args)


if __name__ == "__main__":
    main()