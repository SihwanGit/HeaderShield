"""Paired file-onset versus WebRTC VAD-onset experiment; no bank training."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.experiment-deps'))
sys.path.insert(0, str(ROOT))

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter

import numpy as np
import soundfile as sf
import torch
import torchaudio
import webrtcvad
from utils.header_selector import RuleBasedHeaderSelector


def detect(x, mode):
    # 20 ms frames, first run of three speech frames (60 ms confirmation).
    vad = webrtcvad.Vad(mode)
    pcm = (np.clip(x, -1, 1) * 32767).astype('<i2')
    flags = [vad.is_speech(pcm[i:i+320].tobytes(), 16000)
             for i in range(0, len(pcm)-319, 320)]
    onset = next((i*320 for i in range(len(flags)-2)
                  if all(flags[i:i+3])), None)
    return onset, flags


def silence(flags, start, length):
    selected = [v for i, v in enumerate(flags)
                if start <= i*320 and (i+1)*320 <= start+length]
    return 1-sum(selected)/len(selected) if selected else None


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'analysis/vad_alignment_20261102')
    p.add_argument('--vad-mode', type=int, choices=range(4), default=2)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    selector = RuleBasedHeaderSelector()
    rows, errors = [], []
    files = sorted((ROOT/'dataset').rglob('*.wav'))
    if a.limit:
        files = files[:a.limit]
    for idx, path in enumerate(files):
        try:
            raw, sr = sf.read(str(path), dtype='float32', always_2d=True)
            wave = torch.from_numpy(raw.T.copy())
            if sr != 16000:
                wave = torchaudio.functional.resample(wave, sr, 16000)
            wave = wave.mean(0, keepdim=True).clamp(-1, 1)
            if not wave.numel() or not torch.isfinite(wave).all():
                raise ValueError('empty or nonfinite audio')
            x = wave[0].numpy()
            t = time.perf_counter()
            onset, flags = detect(x, a.vad_mode)
            vad_ms = (time.perf_counter()-t)*1000
            rel = path.relative_to(ROOT/'dataset')
            row = dict(dataset=rel.parts[0], speaker=rel.parts[1], file=str(rel),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       original_sr=sr, duration_s=len(x)/16000,
                       status='no_speech' if onset is None else
                       ('short_aligned' if len(x)-onset < 4800 else 'paired'),
                       onset_s=None if onset is None else onset/16000,
                       confirmation_audio_s=None if onset is None else (onset+960)/16000,
                       aligned_window_available_s=None if onset is None else (onset+4800)/16000,
                       vad_compute_ms=vad_ms)
            for label, start in [('A', 0), ('B', onset)]:
                if start is None:
                    for k in ['header', 'energy', 'zero_crossing_rate', 'spectral_centroid',
                              'analysis_duration', 'silence_ratio', 'feature_ms']:
                        row[label+'_'+k] = None
                    continue
                t = time.perf_counter()
                h, features = selector.select(wave[:, start:], return_features=True)
                row[label+'_feature_ms'] = (time.perf_counter()-t)*1000
                row[label+'_header'] = h
                row.update({label+'_'+k: v for k, v in features.items()})
                row[label+'_silence_ratio'] = silence(flags, start, min(4800, len(x)-start))
            row['changed'] = None if onset is None else row['A_header'] != row['B_header']
            rows.append(row)
        except Exception as e:
            errors.append(dict(file=str(path), error=repr(e)))
        if (idx+1) % 50 == 0:
            print(f'{idx+1}/{len(files)} files', flush=True)
    write_csv(a.output/'utterances.csv', rows)
    write_csv(a.output/'errors.csv', errors)
    summaries = []
    for dataset in sorted({r['dataset'] for r in rows}):
        all_rows = [r for r in rows if r['dataset'] == dataset]
        paired = [r for r in all_rows if r['status'] == 'paired' and r['A_analysis_duration'] == .3]
        s = dict(dataset=dataset, files=len(all_rows), speakers=len({r['speaker'] for r in all_rows}),
                 unique_file_hashes=len({r['sha256'] for r in all_rows}),
                 no_speech=sum(r['status']=='no_speech' for r in all_rows),
                 short_aligned=sum(r['status']=='short_aligned' for r in all_rows), paired_n=len(paired))
        if paired:
            s['changed_percent'] = 100*np.mean([r['changed'] for r in paired])
            s['onset_median_s'] = float(np.median([r['onset_s'] for r in paired]))
            s['vad_compute_mean_ms'] = float(np.mean([r['vad_compute_ms'] for r in paired]))
            for label in ['A', 'B']:
                counts = Counter(r[label+'_header'] for r in paired)
                for h in range(5):
                    s[f'{label}_H{h}_percent'] = counts[h]*100/len(paired)
                s[label+'_H01_percent'] = (counts[0]+counts[1])*100/len(paired)
                s[label+'_entropy_bits'] = -sum((n/len(paired))*math.log2(n/len(paired)) for n in counts.values())
                for feature in ['energy', 'zero_crossing_rate', 'spectral_centroid', 'silence_ratio']:
                    vals = [r[label+'_'+feature] for r in paired]
                    for name, val in [('mean', np.mean(vals)), ('std', np.std(vals)),
                                      ('p25', np.quantile(vals,.25)), ('median', np.median(vals)),
                                      ('p75', np.quantile(vals,.75))]:
                        s[f'{label}_{feature}_{name}'] = float(val)
        summaries.append(s)
    (a.output/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    write_csv(a.output/'summary.csv', summaries)
    transitions = []
    for s in summaries:
        pairs = [r for r in rows if r['dataset']==s['dataset'] and r['status']=='paired']
        for i in range(5):
            for j in range(5):
                transitions.append(dict(dataset=s['dataset'], before=i, after=j,
                                        count=sum(r['A_header']==i and r['B_header']==j for r in pairs)))
    write_csv(a.output/'transitions.csv', transitions)
    speaker_rows = []
    for dataset, speaker in sorted({(r['dataset'], r['speaker']) for r in rows}):
        pairs = [r for r in rows if r['dataset']==dataset and r['speaker']==speaker
                 and r['status']=='paired']
        if pairs:
            speaker_rows.append(dict(dataset=dataset, speaker=speaker, n=len(pairs),
                changed_percent=100*float(np.mean([r['changed'] for r in pairs])),
                A_H01_percent=100*float(np.mean([r['A_header'] in (0,1) for r in pairs])),
                B_H01_percent=100*float(np.mean([r['B_header'] in (0,1) for r in pairs]))))
    write_csv(a.output/'speakers.csv', speaker_rows)
    config = dict(python=platform.python_version(), packages={x:importlib.metadata.version(x)
                  for x in ['torch','torchaudio','numpy','soundfile','webrtcvad-wheels']},
                  selector=vars(selector), selector_sha256=hashlib.sha256((ROOT/'utils/header_selector.py').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  vad_mode=a.vad_mode, frame_ms=20, consecutive_speech_frames=3,
                  sampling='all WAV files; one paired observation per file; no train/test split',
                  silence_definition='fraction of complete 20ms frames classified non-speech by same full-file VAD pass',
                  primary_population='VAD detected and both windows contain full 0.3 seconds',
                  latency_note='VAD compute time is a full-file offline pass, not streaming latency. B requires onset+0.3s audio; confirmation requires onset+0.06s.',
                  errors=errors, command=sys.argv)
    (a.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(summaries), figsize=(13,4), squeeze=False)
    for ax, s in zip(axes[0], summaries):
        bottom = np.zeros(2)
        for h in range(5):
            vals = [s.get(f'{c}_H{h}_percent',0) for c in ['A','B']]
            ax.bar(['File onset','VAD onset'], vals, bottom=bottom, label=f'H{h}')
            bottom += vals
        ax.set_title(s['dataset'].replace('aihub_dataset_', '')+f" (n={s['paired_n']})")
        ax.set_ylim(0,100)
        ax.set_ylabel('Selection (%)')
    axes[0,-1].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(a.output/'selection_distribution.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,4,figsize=(16,4))
    for ax, feature in zip(axes, ['energy','zero_crossing_rate','spectral_centroid','silence_ratio']):
        values, labels = [], []
        for s in summaries:
            pairs = [r for r in rows if r['dataset']==s['dataset'] and r['status']=='paired']
            for label in ['A','B']:
                values.append([r[label+'_'+feature] for r in pairs])
                labels.append(s['dataset'].replace('aihub_dataset_','').replace('_dataset','')+' '+label)
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.tick_params(axis='x', rotation=70)
        ax.set_title(feature)
    fig.tight_layout()
    fig.savefig(a.output/'feature_distributions.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
