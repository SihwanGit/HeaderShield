from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm


def pad_layer(
    inp: Tensor, layer: nn.Module, pad_type: Optional[str] = "reflect"
) -> Tensor:
    """입력을 패딩한 뒤 지정된 계층을 적용
    
    Args:
        inp: 입력 텐서
        layer: 합성곱 계층
        pad_type: 패딩 유형. 기본값은 "reflect"
        
    Returns:
        패딩 후 계층이 적용된 출력
    """
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    inp = F.pad(inp, pad=pad, mode=pad_type)
    out = layer(inp)
    return out


def pixel_shuffle_1d(inp: Tensor, scale_factor: Optional[float] = 2.0) -> Tensor:
    """1차원 픽셀 셔플. 업샘플링에 사용
    
    Args:
        inp: 입력 텐서
        scale_factor: 스케일 계수. 기본값은 2.0
        
    Returns:
        재배열된 텐서
    """
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


def upsample(x: Tensor, scale_factor: Optional[float] = 2.0) -> Tensor:
    """최근접 이웃 보간을 사용한 업샘플링
    
    Args:
        x: 입력 텐서
        scale_factor: 스케일 계수. 기본값은 2.0
        
    Returns:
        업샘플링된 텐서
    """
    x_up = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
    return x_up


def append_cond(x: Tensor, cond: Tensor) -> Tensor:
    """입력에 조건 정보인 평균과 표준편차 적용
    
    Args:
        x: 입력 텐서
        cond: 조건 텐서. 앞쪽 절반은 평균, 뒤쪽 절반은 표준편차
        
    Returns:
        조건이 적용된 텐서
    """
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out


def conv_bank(
    x: Tensor,
    module_list: nn.Module,
    act: nn.Module,
    pad_type: Optional[str] = "reflect",
) -> Tensor:
    """합성곱 뱅크 적용. 여러 크기의 합성곱 커널 사용
    
    Args:
        x: 입력 텐서
        module_list: 합성곱 계층 리스트
        act: 활성화 함수
        pad_type: 패딩 유형. 기본값은 "reflect"
        
    Returns:
        합성곱 뱅크 출력 결과. 모든 합성곱 결과를 연결한 텐서
    """
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out


def get_act(act: str) -> nn.Module:
    """활성화 함수 반환
    
    Args:
        act: 활성화 함수 유형
        
    Returns:
        활성화 함수 모듈
    """
    if act == "lrelu":
        return nn.LeakyReLU()
    return nn.ReLU()


class ContentEncoder(nn.Module):
    """콘텐츠 인코더
    
    입력 음성의 콘텐츠 정보를 추출하며, 화자 특징은 포함하지 않는다.
    """
    def __init__(
        self,
        c_in: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        bank_size: int,
        bank_scale: int,
        c_bank: int,
        n_conv_blocks: int,
        subsample: List[int],
        act: str,
        dropout_rate: float,
    ):
        """콘텐츠 인코더 초기화
        
        Args:
            c_in: 입력 채널 수
            c_h: 은닉층 채널 수
            c_out: 출력 채널 수
            kernel_size: 합성곱 커널 크기
            bank_size: 합성곱 뱅크의 최대 커널 크기
            bank_scale: 합성곱 뱅크 커널 크기 증가 간격
            c_bank: 합성곱 뱅크 채널 수
            n_conv_blocks: 합성곱 블록 개수
            subsample: 각 합성곱 블록의 다운샘플링 비율
            act: 활성화 함수 유형
            dropout_rate: Dropout 비율
        """
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_bank, kernel_size=k)
                for k in range(bank_scale, bank_size + 1, bank_scale)
            ]
        )
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """순전파
        
        Args:
            x: 입력 특징
            
        Returns:
            mu: 평균 벡터
            log_sigma: 로그 표준편차 벡터
        """
        out = conv_bank(x, self.conv_bank, act=self.act)
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        mu = pad_layer(out, self.mean_layer)
        log_sigma = pad_layer(out, self.std_layer)
        return mu, log_sigma


class SpeakerEncoder(nn.Module):
    """화자 인코더
    
    입력 음성의 화자 특징을 추출한다.
    """
    def __init__(
        self,
        c_in: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        bank_size: int,
        bank_scale: int,
        c_bank: int,
        n_conv_blocks: int,
        n_dense_blocks: int,
        subsample: List[int],
        act: str,
        dropout_rate: float,
    ):
        """화자 인코더 초기화
        
        Args:
            c_in: 입력 채널 수
            c_h: 은닉층 채널 수
            c_out: 출력 채널 수
            kernel_size: 합성곱 커널 크기
            bank_size: 합성곱 뱅크의 최대 커널 크기
            bank_scale: 합성곱 뱅크 커널 크기 증가 간격
            c_bank: 합성곱 뱅크 채널 수
            n_conv_blocks: 합성곱 블록 개수
            n_dense_blocks: 완전연결 블록 개수
            subsample: 각 합성곱 블록의 다운샘플링 비율
            act: 활성화 함수 유형
            dropout_rate: Dropout 비율
        """
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_bank, kernel_size=k)
                for k in range(bank_scale, bank_size + 1, bank_scale)
            ]
        )
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.second_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp: Tensor) -> Tensor:
        """합성곱 블록 시퀀스
        
        Args:
            inp: 입력 특징
            
        Returns:
            처리된 특징
        """
        out = inp
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp: Tensor) -> Tensor:
        """완전연결 블록 시퀀스
        
        Args:
            inp: 입력 특징
            
        Returns:
            처리된 특징
        """
        out = inp
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x: Tensor) -> Tensor:
        """순전파
        
        Args:
            x: 입력 특징
            
        Returns:
            화자 임베딩 벡터
        """
        out = conv_bank(x, self.conv_bank, act=self.act)
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        out = self.conv_blocks(out)
        out = self.pooling_layer(out).squeeze(-1)
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out


class Decoder(nn.Module):
    """디코더
    
    콘텐츠 특징과 화자 임베딩을 바탕으로 목표 음성 특징을 생성한다.
    """
    def __init__(
        self,
        c_in: int,
        c_cond: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        n_conv_blocks: int,
        upsample: List[int],
        act: str,
        sn: bool,
        dropout_rate: float,
    ):
        """디코더 초기화
        
        Args:
            c_in: 입력 채널 수
            c_cond: 조건 채널 수
            c_h: 은닉층 채널 수
            c_out: 출력 채널 수
            kernel_size: 합성곱 커널 크기
            n_conv_blocks: 합성곱 블록 개수
            upsample: 각 합성곱 블록의 업샘플링 비율
            act: 활성화 함수 유형
            sn: 스펙트럴 정규화 사용 여부
            dropout_rate: Dropout 비율
        """
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList(
            [
                f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size))
                for _ in range(n_conv_blocks)
            ]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size))
                for _, up in zip(range(n_conv_blocks), self.upsample)
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
            [f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks * 2)]
        )
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        """순전파
        
        Args:
            z: 콘텐츠 특징
            cond: 조건 정보. 화자 임베딩
            
        Returns:
            생성된 음성 특징
        """
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l * 2](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l * 2 + 1](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l])
            else:
                out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out


class AdaInVC(nn.Module):
    """AdaIN-VC 음성 변환 모델
    
    적응형 인스턴스 정규화를 사용해 음성 변환을 수행하는 모델.
    """
    def __init__(self, config: Dict):
        """모델 초기화
        
        Args:
            config: 모델 설정 딕셔너리
        """
        super(AdaInVC, self).__init__()
        self.content_encoder = ContentEncoder(**config["ContentEncoder"])
        self.speaker_encoder = SpeakerEncoder(**config["SpeakerEncoder"])
        self.decoder = Decoder(**config["Decoder"])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """순전파. 학습 모드
        
        Args:
            x: 입력 특징
            
        Returns:
            mu: 콘텐츠 평균
            log_sigma: 콘텐츠 로그 표준편차
            emb: 화자 임베딩
            dec: 디코딩 결과
        """
        mu, log_sigma = self.content_encoder(x)
        emb = self.speaker_encoder(x)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        dec = self.decoder(mu + torch.exp(log_sigma / 2) * eps, emb)
        return mu, log_sigma, emb, dec

    def inference(self, src: Tensor, tgt: Tensor) -> Tensor:
        """추론. 음성 변환
        
        Args:
            src: 원본 음성 특징. 언어 내용을 제공함
            tgt: 목표 음성 특징. 음색 특징을 제공함
            
        Returns:
            변환된 음성 특징
        """
        mu, _ = self.content_encoder(src)
        emb = self.speaker_encoder(tgt)
        dec = self.decoder(mu, emb)
        return dec