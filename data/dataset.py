import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple, Dict, Optional


class VCTKDataset(Dataset):
    def __init__(self, root_dir: str, speaker_id: str, transform=None,
                 split='train', window_size=100, shift_size=50, sample_rate=16000):
        """
        VCTK 데이터셋 로더 클래스
        
        Args:
            root_dir: 데이터셋 루트 디렉터리
            speaker_id: 학습에 사용할 화자 ID
            transform: 데이터 변환 함수
            split: 'train' 또는 'test'
            window_size: 슬라이딩 윈도우 크기
            shift_size: 슬라이딩 윈도우 이동 간격
            sample_rate: 오디오 샘플링 레이트
        """
        self.root_dir = root_dir
        self.speaker_id = speaker_id
        self.transform = transform
        self.window_size = window_size
        self.shift_size = shift_size
        self.sample_rate = sample_rate

        # 수정: speaker_id가 "001"로 들어와도 "p001" 폴더를 찾고,
        #       "p001"로 들어와도 그대로 사용할 수 있도록 처리
        if str(speaker_id).startswith('p'):
            self.speaker_folder = str(speaker_id)
        else:
            self.speaker_folder = f'p{speaker_id}'

        # 해당 화자의 모든 오디오 파일 가져오기
        speaker_dir = os.path.join(root_dir, self.speaker_folder)

        # 수정: 폴더가 없을 경우 원인을 바로 알 수 있도록 명확한 에러 메시지 추가
        if not os.path.isdir(speaker_dir):
            raise FileNotFoundError(
                f"화자 폴더를 찾을 수 없습니다: {speaker_dir}\n"
                f"현재 코드는 다음과 같은 구조를 기대합니다.\n"
                f"  {root_dir}/p001/*.wav\n"
                f"  {root_dir}/p002/*.wav\n"
                f"예시 실행:\n"
                f"  --data_dir ./data/demo_vsmask --target_speaker 001 --other_speakers 002"
            )

        # 수정: .wav뿐 아니라 .WAV도 읽을 수 있도록 lower() 적용
        self.audio_files = [f for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]
        self.audio_files.sort()

        # 수정: wav 파일이 없을 때 바로 원인을 알 수 있도록 에러 추가
        if len(self.audio_files) == 0:
            raise FileNotFoundError(
                f"wav 파일을 찾을 수 없습니다: {speaker_dir}\n"
                f"화자 폴더 안에 .wav 파일이 있는지 확인하세요."
            )

        # 학습 세트와 테스트 세트 분리
        random.seed(42)  # 재현성을 보장하기 위해 랜덤 시드 고정
        random.shuffle(self.audio_files)

        split_idx = int(0.8 * len(self.audio_files))

        if split == 'train':
            self.audio_files = self.audio_files[:split_idx]
        else:
            self.audio_files = self.audio_files[split_idx:]

        # 수정: 샘플 수가 너무 적으면 train/test 중 하나가 비어버릴 수 있으므로 방어 처리
        #       현재 demo_vsmask처럼 파일 수가 적은 smoke test에서 필요함
        if len(self.audio_files) == 0:
            self.audio_files = [f for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]
            self.audio_files.sort()

        # 오디오를 전처리하고 슬라이딩 윈도우 구간 생성
        self.segments = self._preprocess_audio()

        # 수정: 세그먼트가 하나도 만들어지지 않으면 원인을 설명하는 에러 추가
        if len(self.segments) == 0:
            raise RuntimeError(
                f"생성된 오디오 세그먼트가 없습니다.\n"
                f"speaker_dir: {speaker_dir}\n"
                f"wav 파일 수: {len(self.audio_files)}\n"
                f"window_size: {self.window_size} samples "
                f"({self.window_size / self.sample_rate:.3f} sec)\n"
                f"shift_size: {self.shift_size} samples "
                f"({self.shift_size / self.sample_rate:.3f} sec)\n"
                f"원인 후보:\n"
                f"1. wav 길이가 window_size보다 짧음\n"
                f"2. sample_rate 변환 후 길이가 너무 짧음\n"
                f"3. 유효한 wav 파일이 없음\n"
                f"해결:\n"
                f"- smoke test라면 window_size를 줄이세요.\n"
                f"- 또는 더 긴 wav 파일을 사용하세요."
            )

    def _preprocess_audio(self) -> List[Tuple[torch.Tensor, int]]:
        """오디오를 전처리하고 슬라이딩 윈도우 구간 생성"""
        segments = []
        self.segment_files = []

        for audio_file in self.audio_files:
            # 수정: speaker_id 대신 self.speaker_folder 사용
            #       "001"과 "p001" 입력을 모두 지원하기 위함
            file_path = os.path.join(self.root_dir, self.speaker_folder, audio_file)
            waveform, sr = torchaudio.load(file_path)

            # 수정: 빈 wav 파일 방어
            if waveform.numel() == 0:
                continue

            # 목표 샘플링 레이트로 리샘플링
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # 모노 채널로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 수정: waveform 값 범위를 방어적으로 제한
            waveform = torch.clamp(waveform, -1.0, 1.0)

            # 추가 변환 적용. transform이 있는 경우에만 수행
            if self.transform:
                waveform = self.transform(waveform)

            # 수정: wav가 window_size보다 짧으면 zero-padding하여 최소 1개 구간 생성
            #       기존 코드는 짧은 wav에서 segment가 0개가 될 수 있었음
            if waveform.shape[1] < self.window_size:
                pad_len = self.window_size - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))

            # 슬라이딩 윈도우 구간 생성
            # 수정: 기존 range(0, waveform.shape[1] - self.window_size, ...)
            #       형태는 정확히 window_size 길이인 wav에서 구간을 만들지 못함.
            #       +1을 추가해 최소 1개 segment가 생성되도록 수정
            for i in range(0, waveform.shape[1] - self.window_size + 1, self.shift_size):
                segment = waveform[:, i:i+self.window_size]
                segments.append((segment, i))
                self.segment_files.append(os.path.abspath(file_path))

        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        segment, position = self.segments[idx]
        return {
            'waveform': segment,
            'source_file': self.segment_files[idx],
            'slice_id': idx,
            # 수정: DataLoader collate 안정성을 위해 position을 tensor로 반환
            'position': torch.tensor(position, dtype=torch.long)
        }


class MultiSpeakerDataset(Dataset):
    def __init__(self, root_dir: str, target_speaker_id: str, other_speaker_ids: List[str],
                 transform=None, window_size=100, shift_size=50, sample_rate=16000):
        """
        VSMask 학습을 위한 다중 화자 데이터셋
        
        Args:
            root_dir: 데이터셋 루트 디렉터리
            target_speaker_id: 목표 화자 ID. 보호 대상
            other_speaker_ids: 기타 화자 ID 목록. 오도 목적으로 사용
            transform: 데이터 변환 함수
            window_size: 슬라이딩 윈도우 크기
            shift_size: 슬라이딩 윈도우 이동 간격
            sample_rate: 오디오 샘플링 레이트
        """
        self.root_dir = root_dir

        # 수정: other_speaker_ids가 비어 있을 경우 명확한 에러 출력
        if len(other_speaker_ids) == 0:
            raise ValueError("other_speaker_ids는 최소 1개 이상 필요합니다.")

        self.target_dataset = VCTKDataset(root_dir, target_speaker_id, transform,
                                         'train', window_size, shift_size, sample_rate)

        # 기타 화자 중 한 명을 무작위로 선택하여 목표 오도 화자로 사용
        self.other_speaker_id = random.choice(other_speaker_ids)
        self.other_dataset = VCTKDataset(root_dir, self.other_speaker_id, transform,
                                        'train', window_size, shift_size, sample_rate)

    def __len__(self) -> int:
        return len(self.target_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_data = self.target_dataset[idx]

        # 기타 화자의 샘플 중 하나를 무작위로 선택
        other_idx = random.randint(0, len(self.other_dataset) - 1)
        other_data = self.other_dataset[other_idx]

        return {
            'source_waveform': target_data['waveform'],
            'source_position': target_data['position'],
            'source_file': target_data['source_file'],
            'source_slice_id': target_data['slice_id'],
            'target_waveform': other_data['waveform'],
            'target_position': other_data['position'],
            'target_speaker_id': self.other_speaker_id
        }


def get_dataloaders(root_dir: str, target_speaker_id: str, other_speaker_ids: List[str],
                   batch_size: int = 32, window_size: int = 100, shift_size: int = 50,
                   sample_rate: int = 16000, num_workers: int = 0):
    """
    학습 및 테스트 데이터 로더 생성
    
    Args:
        root_dir: 데이터셋 루트 디렉터리
        target_speaker_id: 목표 화자 ID
        other_speaker_ids: 기타 화자 ID 목록
        batch_size: 배치 크기
        window_size: 슬라이딩 윈도우 크기
        shift_size: 슬라이딩 윈도우 이동 간격
        sample_rate: 오디오 샘플링 레이트 (단위는 Hz)
        num_workers: 데이터 로딩 작업 스레드 수

        이중 window_size와 shift_size는 논문 규격인 1.65초, 0.4초에 맞춰 조정함.
        
    Returns:
        train_loader, test_loader
    """
    # 학습 세트 데이터 로더 생성
    train_dataset = MultiSpeakerDataset(
        root_dir, target_speaker_id, other_speaker_ids,
        window_size=window_size, shift_size=shift_size, sample_rate=sample_rate
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        # 수정: WSL/Windows 및 CPU smoke test에서 멀티프로세싱 문제를 줄이기 위해 기본값을 0으로 변경
        num_workers=num_workers,
        # 수정: CPU 환경에서도 문제 없도록 pin_memory=False로 변경
        pin_memory=False
    )

    # 테스트 세트 데이터 로더 생성
    test_dataset = VCTKDataset(
        root_dir, target_speaker_id, split='test',
        window_size=window_size, shift_size=shift_size, sample_rate=sample_rate
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        # 수정: WSL/Windows 및 CPU smoke test에서 멀티프로세싱 문제를 줄이기 위해 기본값을 0으로 변경
        num_workers=num_workers,
        # 수정: CPU 환경에서도 문제 없도록 pin_memory=False로 변경
        pin_memory=False
    )

    return train_loader, test_loader