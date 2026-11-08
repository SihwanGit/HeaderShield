# HeaderShield

HeaderShield is a research implementation for protecting speaker identity using
learned perturbation headers and a condition-dependent header bank. It was
developed from the VSMask codebase; legacy attack and voice conversion utilities
are retained for research compatibility. Protection effectiveness depends on
the evaluated model and data and is not a guarantee against voice cloning.

## Installation

Use a separate Python environment (Python 3.10+ recommended). Install a matching
PyTorch and torchaudio pair for your CPU/CUDA platform, then install the remaining
dependencies:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
```

Dependency versions are not pinned; this is a source release, not a locked
reproduction environment. The lightweight unit tests do not require ML packages.

## Data and pretrained models

Datasets, audio samples, model weights, experiment outputs, and local logs are
excluded. Obtain AIHub/VCTK data from the original providers and comply with
their terms. Training accepts a local dataset path with `--data_dir`; see
`data/dataset.py` for the expected speaker-directory layout.

For RTVC encoder training, obtain the speaker encoder from the
[upstream RTVC project](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
and place it at `external/saved_models/default/encoder.pt`.
The vendored encoder source and its MIT notice are included; weights are not.
Only load trusted PyTorch checkpoint files.

## Training and applying a header bank

Replace the example dataset path and speaker IDs with your local data:

```bash
python train_header_bank_condition_subset_encoder.py --data_dir dataset/my_dataset --target_speaker p001 --other_speakers p002 p003 --external_root external --output_dir output/header_bank
python apply_header_only.py --mode bank --header_bank output/header_bank/header_bank.pt --input path/to/input.wav --output output/protected.wav
```

Use `--help` on each script for options. `train_header.py` trains a universal
header; `train_header_bank.py` and `train_header_bank_condition_subset.py`
provide alternative bank training paths. `prepare_headerbank_evaluation.py`
exports selector/forced-header variants for a separately installed evaluation
pipeline. That external evaluation pipeline is not bundled here.

## Verification

The dependency-free regression tests can be run with:

```bash
python -m unittest discover -s tests -p test_header_training_log.py
python -m unittest discover -s tests -p test_header_protection_selection.py
python -m unittest discover -s tests -p test_prepare_headerbank_evaluation.py
```

The optional `tests/test_vad_alignment.py` requires the ML/VAD dependencies;
its real-speech test skips when the local dataset is absent.

## Publication and attribution

See [PUBLICATION_POLICY.md](PUBLICATION_POLICY.md) for exclusions and historical
Git content, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for original
sources. Model downloads and datasets remain subject to their own terms.
No new project-wide license is granted by this source release.

## Legacy Attack-VC documentation

The following section is retained from the original codebase and describes the
upstream work, rather than the HeaderShield paper.

### Defending Your Voice: Adversarial Attack on Voice Conversion

This is the official implementation of the paper [Defending Your Voice: Adversarial Attack on Voice Conversion](https://arxiv.org/abs/2005.08781).
We use adversarial attack to prevent one's voice from improperly utilized in voice conversion.
The conversion fails when adversarial noise is added to the input utterance.

For the audio samples, please refer to our [demo page](https://yistlin.github.io/attack-vc-demo/).

## Pre-trained Model

The pre-trained [AdaIN-VC](https://arxiv.org/abs/1904.05742) (referred to as Chou's model in the paper) is available [here](https://drive.google.com/drive/folders/1vGU7DdfuebG1y8f4s4MOqEMunzsxxAHX?usp=drive_link).
The files are from the official implementation of AdaIN-VC, but the config file was slightly modified.

## Attack

You can perform adversarial attack on AdaIN-VC with `attack.py`.

```bash
python attack.py <model_dir> <vc_tgt> <adv_tgt> <output> [--vc_src source] [--eps epsilon] [--n_iters iterations] [--attack_type type]
```

- **model_dir**: The directory of model files.
- **vc_tgt**: The target utterance to be defended, providing vocal timbre in voice conversion.
- **adv_tgt**: The target used in adversarial attack (**y** in the paper).
- **output**: The output defended utterance.
- **source**: The source utterance providing linguistic content in voice conversion (required in end-to-end and feedback attack).
- **epsilon**: The maximum amplitude of the perturbation.
- **iterations**: The number of iterations for updating the perturbation.
- **type**: The type of adversarial attack to use (end-to-end, embedding, or feedback attack).

## Inference

You can perform voice conversion with `inference.py`.

```bash
python inference.py <model_dir> <source> <target> <output>
```

- **model_dir**: The directory of model files.
- **source**: The source utterance providing linguistic content in voice conversion.
- **target**: The target utterance providing vocal timbre in voice conversion.
- **output**: The output converted utterance.

## Header Bank 학습 로그

`train_header_bank.py`, `train_header_bank_condition_subset.py`,
`train_header_bank_condition_subset_encoder.py` 실행 시 별도 옵션 없이
`<output_dir>/logs/<실행 시각>/`에 다음 파일이 생성됩니다.

- `slice_assignments.csv`: 슬라이스 ID, 원본 WAV 경로, sample/초 단위 구간,
  Selector 원래 판정(`raw_header`), 뱅크 범위 보정 후 판정(`assigned_header`),
  헤더 이름, RMS energy, ZCR, spectral centroid. 같은 슬라이스는 epoch가 반복돼도 한 번만 기록합니다.
- `training_steps.csv`: optimizer step 성공 후 학습 헤더별 행을 기록합니다.
  분류 헤더와 학습 헤더, primary/borrowed/default 또는 학습 전략,
  원래 가중치와 step 내 정규화 가중치, 헤더 loss 및 전체 step loss를 제공합니다.
  `slice_id`로 분류 CSV와 연결할 수 있습니다.
- `summary.json`: 헤더별 분류 개수·비율, 누적 학습 참여 횟수·비율,
  가중치 합·비율, 평균 헤더 loss. 미학습 헤더의 평균 loss는 `null`입니다.
- `header_assignment_summary.csv`: 헤더별 배정 슬라이스 수(`assigned_slices`),
  전체 분류 슬라이스 수(`total_slices`), 배정 비율(`assigned_percent`, 0~100).
  배정이 0개인 헤더도 표시하며 summary와 같은 시점에 갱신합니다.
- `config.json`: 해당 실행의 학습 설정 및 Selector 임계값.

분류 비율의 분모는 **실제로 분류한 고유 슬라이스 수**입니다. Subset 방식에서는
subset 구성 단계에서 집계되며, `--max_subset_samples`가 있으면 그 한도 내 표본입니다.
기본 `train_header_bank.py`는 학습 중 만난 슬라이스만 집계하므로 `--max_steps`로
일찍 종료하면 전체 데이터셋 분포와 다를 수 있습니다. 겹치는 슬라이스도 각각 한 개로
세므로 이 비율은 음성 길이의 비율이 아닙니다.

학습 비율의 분모는 모든 헤더의 학습 참여 횟수 합입니다. 한 step에서 여러 헤더가
참여할 수 있으며 borrowed/default 보조 학습도 포함됩니다. 이는 optimizer의 모멘텀에
의한 파라미터 변화 측정이 아니라 loss 계산 참여 내역입니다.
콘솔 및 summary는 `--log_interval` step마다, epoch 종료 및 학습 완료 시 갱신되며
모두 실행 시작 이후 누적값입니다. CSV는 출력 간격과 무관하게 매 슬라이스/step 기록됩니다.
중단 시 CSV에는 완료된 기록이 남고 summary는 마지막 갱신 시점의 값입니다.

구간 좌표는 리샘플링 후 기준이고 종료점은 exclusive이며 짧은 WAV의 zero padding을
포함할 수 있습니다. 현재 학습은 슬라이딩 윈도우 앞부분을 분석하므로 실제 VAD로 검출한
발화 시작점과 반드시 일치하지는 않습니다.

## 헤더별 합성 보호 성능 비교

`apply_header_only.py --mode bank`는 보호 WAV 옆에 `.wav.json` 메타데이터를 자동 저장합니다.
`--force_header_index 0`~`4`를 지정하면 해당 헤더를 강제로 적용하고, 생략하면 Selector가 선택합니다.
`prepare_headerbank_evaluation.py --bank-root <학습결과> --output-root <새 실험 입력 폴더>`로
원본·자동 선택·헤더 0~4 변형을 일괄 생성할 수 있습니다. 선택적으로 `--universal-root`를 지정합니다.
AntiFake의 `HEADER_BANK_EVALUATION.md`에 생성→합성→헤더별 유사도 감소율 평가 명령과 열 설명이 있습니다.

## Reference

Please cite our paper if you find it useful.

```bib
@INPROCEEDINGS{9383529,
  author={C. -y. {Huang} and Y. Y. {Lin} and H. -y. {Lee} and L. -s. {Lee}},
  booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)},
  title={Defending Your Voice: Adversarial Attack on Voice Conversion},
  year={2021},
  volume={},
  number={},
  pages={552-559},
  doi={10.1109/SLT48900.2021.9383529}}
```
# 평가 로그의 데이터셋 구분

`prepare_headerbank_evaluation.py`에 `--dataset-id 데이터셋명 --split test`를 지정하면
각 보호 WAV의 JSON에 데이터셋과 분할이 저장됩니다. 단일 파일을 적용하는
`apply_header_only.py`에서는 `--dataset_id`와 `--split`을 사용합니다.
AntiFake로 WAV와 JSON을 함께 전달하면 적용 헤더 이력, 헤더별 보호 성능,
동일 표본의 selector 대 모든 강제 헤더 비교 및 임계값별 보고서를 생성할 수 있습니다.
실행·해석 방법은 AntiFake의 `HEADER_BANK_EVALUATION.md`를 참고하세요.

