"""Export source, selector and forced-header variants for AntiFake evaluation."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent / "aihub_dataset")
    parser.add_argument("--bank-root", type=Path, required=True, help="Directory containing girl1/header_bank.pt, etc.")
    parser.add_argument("--output-root", type=Path, required=True, help="Use a new experiment input directory")
    parser.add_argument("--universal-root", type=Path, help="Optional directory containing girl1/universal_header.pt, etc.")
    parser.add_argument("--speakers", nargs="+", default=["girl1", "girl2", "girl3", "man1", "man2", "man3"])
    parser.add_argument("--header-indices", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-id", default="", help="Defaults to source-root directory name")
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--limit-files", type=int, default=0, help="0 = all files per speaker")
    parser.add_argument("--dry-run", action="store_true")
    args, protection_options = parser.parse_known_args()
    if any(index < 0 for index in args.header_indices) or len(set(args.header_indices)) != len(args.header_indices):
        parser.error("Header indices must be distinct nonnegative integers")
    if args.limit_files < 0:
        parser.error("--limit-files must be nonnegative")
    # Extra flags are forwarded only for audio/selector settings; never override paths/mode.
    allowed = {"--sample_rate", "--n_fft", "--hop_length", "--n_mels", "--num_headers", "--header_length",
               "--analysis_duration", "--low_energy_threshold", "--high_energy_threshold",
               "--high_centroid_threshold", "--high_zcr_threshold", "--noisy_zcr_threshold"}
    if len(protection_options) % 2 or any(flag not in allowed for flag in protection_options[::2]):
        parser.error("Unknown protection options; use --option value for audio/selector settings")
    jobs, copies = [], []
    script = Path(__file__).resolve().parent / "apply_header_only.py"
    for speaker in args.speakers:
        source_dir = args.source_root / ("p" + speaker)
        if not source_dir.is_dir():
            source_dir = args.source_root / speaker
        files = sorted(source_dir.glob("*.wav"))
        if args.limit_files:
            files = files[:args.limit_files]
        bank = args.bank_root / speaker / "header_bank.pt"
        if not files or not bank.is_file():
            parser.error("Missing source WAVs or bank for {}".format(speaker))
        for source in files:
            copies.append((source, args.output_root / "source" / ("p" + speaker) / source.name))
            variants = [("protected_wavs_aihub", ["--mode", "bank", "--header_bank", str(bank)])]
            variants += [("header_bank_h{}".format(index), ["--mode", "bank", "--header_bank", str(bank),
                          "--force_header_index", str(index)]) for index in args.header_indices]
            if args.universal_root:
                fixed = args.universal_root / speaker / "universal_header.pt"
                if not fixed.is_file():
                    parser.error("Missing universal header: {}".format(fixed))
                variants.append(("universal_protected", ["--mode", "fixed", "--header", str(fixed)]))
            for folder, options in variants:
                output = args.output_root / folder / speaker / ("protected" + source.name)
                command = [sys.executable, str(script), "--input", str(source), "--output", str(output),
                           "--seed", str(args.seed), "--dataset_id", args.dataset_id or args.source_root.resolve().name,
                           "--split", args.split] + options + protection_options
                if args.device:
                    command += ["--device", args.device]
                jobs.append((command, output))
    for output in [p for _, p in copies] + [p for _, p in jobs]:
        if output.exists() or Path(str(output) + ".json").exists():
            parser.error("Output exists; choose a new --output-root: {}".format(output))
    for source, output in copies:
        if not args.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
    for command, output in jobs:
        print("[protect] {}".format(output), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
