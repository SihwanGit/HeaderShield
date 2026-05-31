import os
import argparse
from typing import Tuple

import torch
import torchaudio
import tqdm

from models.predictive_model import PredictiveModel
from models.header_bank import HeaderBank
from utils.audio import MelSpectrogramConverter, match_waveform_length
from utils.header_selector import RuleBasedHeaderSelector


def crop_perturbation_to_mel(predicted_perturbation: torch.Tensor,
                             reference_mel: torch.Tensor) -> torch.Tensor:
    """
    PredictiveModel의 출력 주파수/시간 크기가 reference_mel과 정확히 일치하지 않을 수 있다.
    inference 단계에서 shape mismatch를 피하기 위해 적용 가능한 범위로 crop한다.

    Args:
        predicted_perturbation: 예측된 교란 [B, 1, Fp, Tp]
        reference_mel: 기준 멜 스펙트로그램 [B, 1, Fr, Tr]

    Returns:
        reference_mel에 적용 가능한 크기로 잘린 교란 [B, 1, min(Fp, Fr), min(Tp, Tr)]
    """
    freq_len = min(reference_mel.shape[2], predicted_perturbation.shape[2])
    time_len = min(reference_mel.shape[3], predicted_perturbation.shape[3])

    return predicted_perturbation[:, :, :freq_len, :time_len]


class HeaderBankVSMask:
    def __init__(
        self,
        predictive_model_path: str,
        header_bank_path: str,
        device: str = "cuda",
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        num_headers: int = 5,
        header_length: int = 100,
        analysis_duration: float = 0.3,
        low_energy_threshold: float = 0.01,
        high_energy_threshold: float = 0.08,
        high_centroid_threshold: float = 3000.0,
        high_zcr_threshold: float = 0.18,
        noisy_zcr_threshold: float = 0.25,
    ):
        """
        Header Bank 기반 VSMask inference 클래스.

        기존 vsmask.py는 fixed universal header 하나를 사용하지만,
        이 클래스는 입력 wav의 발화 시작 조건을 selector로 분석한 뒤
        Header Bank에서 적절한 header를 선택해 적용한다.

        Args:
            predictive_model_path: 학습된 predictive model 경로
            header_bank_path: 학습된 Header Bank 경로
            device: 사용할 장치
            sample_rate: 오디오 샘플링 레이트
            n_fft: FFT 크기
            hop_length: mel frame 이동 간격
            n_mels: mel bin 개수
            num_headers: Header Bank의 header 개수
            header_length: header 길이. mel frame 단위
            analysis_duration: selector가 분석할 발화 시작 구간 길이. 초 단위
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA를 사용할 수 없습니다. device를 cpu로 변경합니다.")
            device = "cpu"

        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_headers = num_headers
        self.header_length = header_length

        self.converter = MelSpectrogramConverter(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # 수정: utils/audio.py에 to() 메서드를 추가한 경우 해당 메서드를 사용
        if hasattr(self.converter, "to"):
            self.converter = self.converter.to(device)
        else:
            self.converter.mel_transform = self.converter.mel_transform.to(device)
            self.converter.inverse_mel_transform = self.converter.inverse_mel_transform.to(device)
            self.converter.griffin_lim = self.converter.griffin_lim.to(device)

        self.predictive_model = PredictiveModel(
            mel_bins=n_mels,
            time_dim=78,
            output_time_dim=25,
        ).to(device)

        self.predictive_model.load_state_dict(
            torch.load(predictive_model_path, map_location=device)
        )
        self.predictive_model.eval()

        self.header_bank = HeaderBank(
            num_headers=num_headers,
            mel_bins=n_mels,
            time_length=header_length,
            device=device,
        )
        self.header_bank.load(header_bank_path)

        self.selector = RuleBasedHeaderSelector(
            sample_rate=sample_rate,
            analysis_duration=analysis_duration,
            low_energy_threshold=low_energy_threshold,
            high_energy_threshold=high_energy_threshold,
            high_centroid_threshold=high_centroid_threshold,
            high_zcr_threshold=high_zcr_threshold,
            noisy_zcr_threshold=noisy_zcr_threshold,
        )

        print("Header Bank VSMask 시스템이 초기화되었습니다.")
        print("[설정 확인]")
        print(f"  device           : {self.device}")
        print(f"  sample_rate      : {self.sample_rate}")
        print(f"  n_fft            : {self.n_fft}")
        print(f"  hop_length       : {self.hop_length}")
        print(f"  n_mels           : {self.n_mels}")
        print(f"  num_headers      : {self.header_bank.num_headers}")
        print(f"  header_length    : {self.header_bank.time_length}")
        print(f"  analysis_duration: {analysis_duration}")

    def load_waveform(self, input_path: str) -> Tuple[torch.Tensor, int]:
        """
        wav 파일을 로드하고 모델 입력에 맞게 전처리한다.

        Args:
            input_path: 입력 wav 파일 경로

        Returns:
            waveform: mono waveform [1, T]
            sample_rate: 저장에 사용할 sample rate
        """
        waveform, sample_rate = torchaudio.load(input_path)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.sample_rate

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = waveform.to(self.device)

        return waveform, sample_rate

    def apply_selected_header(self,
                              mel_spec: torch.Tensor,
                              waveform: torch.Tensor,
                              verbose: bool = True) -> Tuple[torch.Tensor, int]:
        """
        selector가 고른 header를 mel-spectrogram 앞부분에 적용한다.

        Args:
            mel_spec: 입력 mel-spectrogram [B, 1, F, T]
            waveform: 입력 waveform [1, T]
            verbose: selector 결과 출력 여부

        Returns:
            perturbed_mel: 선택된 header가 적용된 mel
            header_index: 선택된 header index
        """
        header_index, features = self.selector.select(
            waveform.detach().cpu(),
            return_features=True,
        )

        if header_index >= self.header_bank.num_headers:
            header_index = self.header_bank.num_headers - 1

        if verbose:
            print("[Selector 결과]")
            print(f"  selected_header_index: {header_index}")
            print(f"  energy               : {features['energy']:.6f}")
            print(f"  zero_crossing_rate   : {features['zero_crossing_rate']:.6f}")
            print(f"  spectral_centroid    : {features['spectral_centroid']:.2f}")
            print(f"  analysis_duration    : {features['analysis_duration']:.3f}")

        perturbed_mel = self.header_bank.apply_header(mel_spec, header_index)

        return perturbed_mel, header_index

    def apply_predictive_perturbation(
        self,
        original_mel: torch.Tensor,
        perturbed_mel: torch.Tensor,
        window_size: int = 78,
        future_step: int = 25,
        epsilon1: float = 0.115,
        epsilon2: float = 0.085,
        epsilon3: float = 0.1,
    ) -> torch.Tensor:
        """
        predictive model을 이용해 header 이후 구간에 실시간 예측 교란을 적용한다.

        Args:
            original_mel: 원본 mel [B, 1, F, T]
            perturbed_mel: header가 적용된 mel [B, 1, F, T]
            window_size: predictive model 입력 window 크기. mel frame 단위
            future_step: sliding step. mel frame 단위
            epsilon1: 저주파 perturbation constraint
            epsilon2: 중주파 perturbation constraint
            epsilon3: 고주파 perturbation constraint

        Returns:
            predictive perturbation까지 적용된 mel [B, 1, F, T]
        """
        if original_mel.shape[-1] <= window_size:
            return perturbed_mel

        loop_range = range(0, original_mel.shape[-1] - window_size + 1, future_step)

        for start_idx in tqdm.tqdm(loop_range, desc="Applying predictive perturbation"):
            window = original_mel[:, :, :, start_idx:start_idx + window_size]

            with torch.no_grad():
                perturbation = self.predictive_model(window)

            perturbation = crop_perturbation_to_mel(perturbation, original_mel)

            future_idx = start_idx + window_size
            future_end = min(future_idx + perturbation.shape[-1], perturbed_mel.shape[-1])

            if future_idx >= perturbed_mel.shape[-1]:
                continue

            freq_len = min(perturbed_mel.shape[2], perturbation.shape[2])
            apply_time_len = future_end - future_idx

            if apply_time_len <= 0:
                continue

            perturbed_mel[:, :, :freq_len, future_idx:future_end] += \
                perturbation[:, :, :freq_len, :apply_time_len]

        # 수정: utils/audio.py의 apply_weighted_constraint가 3D/4D를 모두 지원하도록 수정되어 있으므로 재사용
        weighted_perturbation = self.converter.apply_weighted_constraint(
            perturbed_mel - original_mel,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
        )

        perturbed_mel = original_mel + weighted_perturbation
        perturbed_mel = torch.clamp(perturbed_mel, -1.0, 1.0)

        return perturbed_mel

    def protect_file(
        self,
        input_path: str,
        output_path: str,
        window_size: int = 78,
        future_step: int = 25,
        epsilon1: float = 0.115,
        epsilon2: float = 0.085,
        epsilon3: float = 0.1,
        use_predictive: bool = True,
    ) -> None:
        """
        Header Bank + Selector 기반으로 wav 파일을 보호한다.

        Args:
            input_path: 입력 wav 파일 경로
            output_path: 출력 wav 파일 경로
            window_size: predictive model 입력 window 크기. mel frame 단위
            future_step: predictive perturbation 적용 step. mel frame 단위
            epsilon1: 저주파 perturbation constraint
            epsilon2: 중주파 perturbation constraint
            epsilon3: 고주파 perturbation constraint
            use_predictive: True이면 predictive model까지 적용, False이면 선택된 header만 적용
        """
        print("[보호 시작]")
        print(f"  input_path    : {input_path}")
        print(f"  output_path   : {output_path}")
        print(f"  window_size   : {window_size} mel frames ({window_size * self.hop_length / self.sample_rate:.3f} sec)")
        print(f"  future_step   : {future_step} mel frames ({future_step * self.hop_length / self.sample_rate:.3f} sec)")
        print(f"  use_predictive: {use_predictive}")

        waveform, sample_rate = self.load_waveform(input_path)
        original_length = waveform.shape[-1]

        mel_spec = self.converter.waveform_to_mel(waveform)

        # 수정: waveform_to_mel은 [1, F, T]를 반환하므로 model/header 입력에 맞게 [B, 1, F, T]로 변환
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)

        perturbed_mel, header_index = self.apply_selected_header(
            mel_spec,
            waveform,
            verbose=True,
        )

        if use_predictive:
            perturbed_mel = self.apply_predictive_perturbation(
                original_mel=mel_spec,
                perturbed_mel=perturbed_mel,
                window_size=window_size,
                future_step=future_step,
                epsilon1=epsilon1,
                epsilon2=epsilon2,
                epsilon3=epsilon3,
            )

        protected_waveform = self.converter.mel_to_waveform(perturbed_mel)
        protected_waveform = match_waveform_length(protected_waveform, original_length)
        protected_waveform = torch.clamp(protected_waveform, -1.0, 1.0)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        torchaudio.save(output_path, protected_waveform.cpu(), sample_rate)

        print("[보호 완료]")
        print(f"  selected_header_index: {header_index}")
        print(f"  saved_to             : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Header Bank + Selector 기반 VSMask inference")

    # 모델 인자
    parser.add_argument("--predictive_model", type=str, required=True,
                        help="예측 모델 경로")
    parser.add_argument("--header_bank", type=str, required=True,
                        help="Header Bank 경로")

    # 오디오 파일 인자
    parser.add_argument("--input", type=str, required=True,
                        help="입력 오디오 파일 경로")
    parser.add_argument("--output", type=str, required=True,
                        help="출력 오디오 파일 경로")

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

    # 보호 인자
    parser.add_argument("--window_size", type=int, default=78,
                        help="predictive model 입력 window 크기. mel frame 단위. 기본값 78은 약 1.25초")
    parser.add_argument("--future_step", type=int, default=25,
                        help="predictive perturbation 적용 step. mel frame 단위. 기본값 25는 약 0.4초")
    parser.add_argument("--epsilon1", type=float, default=0.115,
                        help="저주파 교란의 최대 크기")
    parser.add_argument("--epsilon2", type=float, default=0.085,
                        help="중주파 교란의 최대 크기")
    parser.add_argument("--epsilon3", type=float, default=0.1,
                        help="고주파 교란의 최대 크기")
    parser.add_argument("--no_predictive", action="store_true",
                        help="지정하면 predictive model을 적용하지 않고 Header Bank만 적용")

    # 기타 인자
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="사용할 장치")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. device를 cpu로 변경합니다.")
        args.device = "cpu"

    model = HeaderBankVSMask(
        predictive_model_path=args.predictive_model,
        header_bank_path=args.header_bank,
        device=args.device,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        num_headers=args.num_headers,
        header_length=args.header_length,
        analysis_duration=args.analysis_duration,
        low_energy_threshold=args.low_energy_threshold,
        high_energy_threshold=args.high_energy_threshold,
        high_centroid_threshold=args.high_centroid_threshold,
        high_zcr_threshold=args.high_zcr_threshold,
        noisy_zcr_threshold=args.noisy_zcr_threshold,
    )

    model.protect_file(
        input_path=args.input,
        output_path=args.output,
        window_size=args.window_size,
        future_step=args.future_step,
        epsilon1=args.epsilon1,
        epsilon2=args.epsilon2,
        epsilon3=args.epsilon3,
        use_predictive=not args.no_predictive,
    )


if __name__ == "__main__":
    main()