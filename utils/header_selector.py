import torch
import torchaudio
from typing import Dict, Optional


class RuleBasedHeaderSelector:
    def __init__(
        self,
        sample_rate: int = 16000,
        analysis_duration: float = 0.3,
        low_energy_threshold: float = 0.01,
        high_energy_threshold: float = 0.08,
        high_centroid_threshold: float = 3000.0,
        high_zcr_threshold: float = 0.18,
        noisy_zcr_threshold: float = 0.25,
    ):
        """
        Rule-based Header Selector
        룰 기반 헤더 셀렉터 코드

        발화 시작 구간의 간단한 음향 특징을 기반으로 Header Bank에서 사용할 header index를 선택한다.

        Header index 설계:
            0: 기본/default header
            1: 저에너지 시작 구간용 header
            2: 고에너지/갑작스러운 시작 구간용 header
            3: spectral centroid가 높은 시작 구간용 header. 무성음/마찰음 계열 대응
            4: zero-crossing rate가 매우 높은 잡음 가능 시작 구간용 header

        Args:
            sample_rate: 입력 waveform의 sample rate
            analysis_duration: 발화 시작 특징을 분석할 길이. 초 단위
            low_energy_threshold: 저에너지 판정 threshold
            high_energy_threshold: 고에너지 판정 threshold
            high_centroid_threshold: 높은 spectral centroid 판정 threshold
            high_zcr_threshold: 무성음/마찰음 계열 판정용 zero-crossing rate threshold
            noisy_zcr_threshold: 잡음 가능 구간 판정용 zero-crossing rate threshold
        """
        self.sample_rate = sample_rate
        self.analysis_duration = analysis_duration
        self.low_energy_threshold = low_energy_threshold
        self.high_energy_threshold = high_energy_threshold
        self.high_centroid_threshold = high_centroid_threshold
        self.high_zcr_threshold = high_zcr_threshold
        self.noisy_zcr_threshold = noisy_zcr_threshold

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        입력 waveform을 [1, T] 형태로 정리한다.

        Args:
            waveform: [T], [1, T], [C, T] 중 하나

        Returns:
            정리된 mono waveform [1, T]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 2:
            raise ValueError(
                f"waveform은 [T] 또는 [C, T] 형태여야 합니다. 현재 shape: {tuple(waveform.shape)}"
            )

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

    def _get_start_region(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        발화 시작 구간 일부를 잘라낸다.

        Args:
            waveform: mono waveform [1, T]

        Returns:
            시작 분석 구간 [1, T_start]
        """
        num_samples = int(self.sample_rate * self.analysis_duration)
        num_samples = max(1, num_samples)

        if waveform.shape[-1] < num_samples:
            return waveform

        return waveform[:, :num_samples]

    def compute_energy(self, waveform: torch.Tensor) -> float:
        """
        RMS energy를 계산한다.

        Args:
            waveform: [1, T]

        Returns:
            RMS energy
        """
        energy = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
        return float(energy.item())

    def compute_zero_crossing_rate(self, waveform: torch.Tensor) -> float:
        """
        Zero-crossing rate를 계산한다.

        Args:
            waveform: [1, T]

        Returns:
            zero-crossing rate
        """
        x = waveform.squeeze(0)

        if x.numel() < 2:
            return 0.0

        signs = torch.sign(x)
        signs[signs == 0] = 1

        crossings = (signs[1:] != signs[:-1]).float()
        zcr = torch.mean(crossings)

        return float(zcr.item())

    def compute_spectral_centroid(self, waveform: torch.Tensor) -> float:
        """
        spectral centroid를 계산한다.

        Args:
            waveform: [1, T]

        Returns:
            spectral centroid, Hz 단위
        """
        if waveform.shape[-1] < 2:
            return 0.0

        n_fft = min(1024, waveform.shape[-1])

        # n_fft가 너무 작을 때도 동작하게 2 이상으로 보정
        n_fft = max(2, n_fft)

        spec = torch.fft.rfft(waveform.squeeze(0), n=n_fft)
        magnitude = torch.abs(spec)

        freqs = torch.linspace(
            0,
            self.sample_rate / 2,
            magnitude.shape[-1],
            device=waveform.device,
        )

        centroid = torch.sum(freqs * magnitude) / (torch.sum(magnitude) + 1e-8)

        return float(centroid.item())

    def extract_features(self, waveform: torch.Tensor) -> Dict[str, float]:
        """
        발화 시작 구간에서 selector에 필요한 특징을 추출한다.

        Args:
            waveform: 입력 waveform [T] 또는 [C, T]

        Returns:
            feature dictionary
        """
        waveform = self._prepare_waveform(waveform)
        start_region = self._get_start_region(waveform)

        energy = self.compute_energy(start_region)
        zcr = self.compute_zero_crossing_rate(start_region)
        centroid = self.compute_spectral_centroid(start_region)

        return {
            "energy": energy,
            "zero_crossing_rate": zcr,
            "spectral_centroid": centroid,
            "analysis_duration": min(
                self.analysis_duration,
                start_region.shape[-1] / self.sample_rate,
            ),
        }

    def select_header_index(self, waveform: torch.Tensor) -> int:
        """
        발화 시작 특징에 따라 Header Bank에서 사용할 header index를 선택한다.

        우선순위:
            1. zcr이 매우 높으면 잡음 가능 구간으로 보고 header 4
            2. spectral centroid와 zcr이 높으면 무성음/마찰음 계열로 보고 header 3
            3. energy가 높으면 갑작스러운 시작으로 보고 header 2
            4. energy가 낮으면 저에너지 시작으로 보고 header 1
            5. 나머지는 기본 header 0

        Args:
            waveform: 입력 waveform [T] 또는 [C, T]

        Returns:
            선택된 header index
        """
        features = self.extract_features(waveform)

        energy = features["energy"]
        zcr = features["zero_crossing_rate"]
        centroid = features["spectral_centroid"]

        if zcr >= self.noisy_zcr_threshold:
            return 4

        if centroid >= self.high_centroid_threshold and zcr >= self.high_zcr_threshold:
            return 3

        if energy >= self.high_energy_threshold:
            return 2

        if energy <= self.low_energy_threshold:
            return 1

        return 0

    def select(self, waveform: torch.Tensor, return_features: bool = False):
        """
        header index를 선택한다.

        Args:
            waveform: 입력 waveform [T] 또는 [C, T]
            return_features: True이면 추출된 feature도 함께 반환

        Returns:
            header_index 또는 (header_index, features)
        """
        features = self.extract_features(waveform)

        energy = features["energy"]
        zcr = features["zero_crossing_rate"]
        centroid = features["spectral_centroid"]

        if zcr >= self.noisy_zcr_threshold:
            header_index = 4
        elif centroid >= self.high_centroid_threshold and zcr >= self.high_zcr_threshold:
            header_index = 3
        elif energy >= self.high_energy_threshold:
            header_index = 2
        elif energy <= self.low_energy_threshold:
            header_index = 1
        else:
            header_index = 0

        if return_features:
            return header_index, features

        return header_index


def load_audio_for_selector(
    audio_path: str,
    sample_rate: int = 16000,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    selector 테스트용 오디오 로더.

    Args:
        audio_path: wav 파일 경로
        sample_rate: 목표 sample rate
        device: 사용할 device. None이면 현재 tensor 기본 device 사용

    Returns:
        mono waveform [1, T]
    """
    waveform, sr = torchaudio.load(audio_path)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = torch.clamp(waveform, -1.0, 1.0)

    if device is not None:
        waveform = waveform.to(device)

    return waveform


def debug_select_header(
    audio_path: str,
    sample_rate: int = 16000,
    analysis_duration: float = 0.3,
) -> None:
    """
    특정 wav 파일에 대해 selector가 어떤 header를 고르는지 확인하는 디버그 함수.

    Args:
        audio_path: wav 파일 경로
        sample_rate: 목표 sample rate
        analysis_duration: 시작 구간 분석 길이
    """
    waveform = load_audio_for_selector(audio_path, sample_rate=sample_rate)

    selector = RuleBasedHeaderSelector(
        sample_rate=sample_rate,
        analysis_duration=analysis_duration,
    )

    header_index, features = selector.select(waveform, return_features=True)

    print(f"audio_path: {audio_path}")
    print(f"selected_header_index: {header_index}")
    print("features:")
    for key, value in features.items():
        print(f"  {key}: {value}")