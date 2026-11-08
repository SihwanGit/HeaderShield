"""Persistent slice assignments and training statistics (no ML dependencies)."""
import csv
import json
from datetime import datetime
from pathlib import Path


HEADER_NAMES = ("default", "low-energy", "high-energy/abrupt", "fricative-like", "noisy-like")


def write_assignment_summary(directory, counts):
    """Write all headers, including empty ones; percentages use unique classified slices."""
    total = sum(counts)
    path = Path(directory) / "header_assignment_summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "header", "header_name", "assigned_slices", "total_slices", "assigned_percent"])
        writer.writeheader()
        for index, count in enumerate(counts):
            writer.writerow(dict(
                header=index,
                header_name=HEADER_NAMES[index] if index < len(HEADER_NAMES) else "extra",
                assigned_slices=count, total_slices=total,
                assigned_percent=round(100 * count / total, 4) if total else 0.0))
    temporary.replace(path)
    return path


def slice_metadata(batch, index, sample_rate):
    """Positions refer to the resampled waveform; end is exclusive, including padding."""
    start = int(batch["source_position"][index])
    length = int(batch["source_waveform"][index].shape[-1])
    return {
        "slice_id": int(batch["source_slice_id"][index]),
        "source_file": batch["source_file"][index],
        "start_sample": start,
        "end_sample": start + length,
        "start_sec": start / sample_rate,
        "end_sec": (start + length) / sample_rate,
    }


class HeaderTrainingLog:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.output_dir) / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.seen = set()
        self.assigned = [0] * args.num_headers
        self.updates = [0] * args.num_headers
        self.weights = [0.0] * args.num_headers
        self.losses = [0.0] * args.num_headers
        self.steps = 0
        self.assignment_fields = ["slice_id", "source_file", "start_sample", "end_sample", "start_sec", "end_sec", "raw_header", "assigned_header", "header_name", "energy", "zero_crossing_rate", "spectral_centroid"]
        self.update_fields = ["epoch", "step", "slice_id", "assigned_header", "trained_header", "sample_type", "weight", "normalized_weight", "header_loss", "step_loss"]
        for name, fields in (("slice_assignments.csv", self.assignment_fields), ("training_steps.csv", self.update_fields)):
            with (self.directory / name).open("w", newline="", encoding="utf-8-sig") as stream:
                csv.DictWriter(stream, fieldnames=fields).writeheader()
        with (self.directory / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(vars(args), stream, ensure_ascii=False, indent=2, default=str)
        print(f"[Header logs] {self.directory.resolve()}")

    def _append(self, name, fields, rows):
        # Close each append so completed records survive training exceptions.
        with (self.directory / name).open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerows(rows)

    def assignment(self, metadata, raw_header, assigned_header, features):
        key = (metadata["source_file"], metadata["slice_id"])
        if key in self.seen:
            return
        self.seen.add(key)
        self.assigned[assigned_header] += 1
        row = dict(metadata, raw_header=raw_header, assigned_header=assigned_header,
                   header_name=self.name(assigned_header))
        row.update({key: float(features[key]) for key in ("energy", "zero_crossing_rate", "spectral_centroid")})
        self._append("slice_assignments.csv", self.assignment_fields, [row])

    @staticmethod
    def name(index):
        return HEADER_NAMES[index] if index < len(HEADER_NAMES) else "extra"

    def training(self, epoch, step, metadata, assigned_header, details, step_loss):
        """Call after optimizer.step; details are (header, weight, loss, sample_type)."""
        total_weight = sum(item[1] for item in details)
        rows = []
        for header, weight, loss, kind in details:
            self.updates[header] += 1
            self.weights[header] += weight
            self.losses[header] += loss
            rows.append(dict(epoch=epoch, step=step, slice_id=metadata["slice_id"],
                             assigned_header=assigned_header, trained_header=header,
                             sample_type=kind, weight=weight, normalized_weight=weight / total_weight,
                             header_loss=loss, step_loss=step_loss))
        self.steps = step
        self._append("training_steps.csv", self.update_fields, rows)

    def summary(self, stage, epoch=0):
        write_assignment_summary(self.directory, self.assigned)
        assigned_total = sum(self.assigned)
        update_total = sum(self.updates)
        weight_total = sum(self.weights)
        rows = []
        for index, count in enumerate(self.assigned):
            rows.append(dict(header=index, name=self.name(index), assigned_slices=count,
                             assigned_percent=100 * count / assigned_total if assigned_total else 0.0,
                             updates=self.updates[index],
                             update_percent=100 * self.updates[index] / update_total if update_total else 0.0,
                             weighted_updates=self.weights[index],
                             weighted_percent=100 * self.weights[index] / weight_total if weight_total else 0.0,
                             mean_loss=self.losses[index] / self.updates[index] if self.updates[index] else None))
        result = dict(stage=stage, epoch=epoch, completed_steps=self.steps,
                      unique_classified_slices=assigned_total, header_update_events=update_total,
                      headers=rows)
        temporary = self.directory / "summary.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "summary.json")
        print(f"[Header distribution: {stage}] classified slices={assigned_total}, update events={update_total} (cumulative)")
        for row in rows:
            mean_loss = "N/A" if row["mean_loss"] is None else f'{row["mean_loss"]:.6f}'
            print(f'  h{row["header"]} {row["name"]}: assigned={row["assigned_slices"]} ({row["assigned_percent"]:.2f}%), '
                  f'updates={row["updates"]} ({row["update_percent"]:.2f}%), mean_loss={mean_loss}')
