#!/usr/bin/env python3
"""Genetic sex per BAM from sex-chromosome read counts (R_Y and R_X), plus a summarize mode.

Per sample:   python src/comp/compute_sex.py <bam> <out_csv>
              -> one row: per-chromosome counts, R_Y, R_X (with 95% CIs), the two calls, sex_call
Summarize:    python src/comp/compute_sex.py --summarize <per_sample_dir> <out_csv>
              -> one row per sample, sorted

Counts: reads on chr1-22, chrX, chrY with MAPQ > 30 (samtools -q 31, the bin.py read
filter), excluding unmapped, secondary, supplementary, QC-fail and duplicate reads and
read 2 of a pair (-F 0xF84), so a paired fragment counts once. The BAMs rebuilt from the
FinaleDB fragment files keep their mapping qualities, so idxstats counts would include
the MAPQ-0 reads shared between chrX and chrY.

R_Y (Skoglund et al. 2013, J Archaeol Sci 40:4477):
    R_Y = n_Y / (n_X + n_Y), 95% CI = R_Y +- 1.96 * sqrt(R_Y (1 - R_Y) / (n_X + n_Y));
    male if the lower bound > 0.077, female if the upper bound < 0.016, else unassigned.
R_X (Mittnik et al. 2016, PLoS ONE 11:e0163019):
    R_X = mean over the 22 autosomes i of (n_X / L_X) / (n_i / L_i), 95% CI = R_X +- 1.96 * SD / sqrt(22);
    female if the lower bound > 0.8, male if the upper bound < 0.6, else unassigned.
sex_call = the shared call when both agree, otherwise "unassigned". The published
thresholds were set on hg19 ancient-DNA data; validate against Cristiano's recorded sex.
(Checked: every assigned Cristiano model sample agrees with Table S1. The
exporters resolve the unassigned ones with src/figures/export/_common.resolve_sex_call:
the single conclusive metric, else male if R_X < 0.8 - again identical to Table S1.)
"""
import glob
import math
import os
import sys

from compute_mean_coverage import extract_sample_name

MAPQ_MIN = "31"        # samtools -q is >=, so 31 == bin.py's MAPQ > 30
EXCLUDE_FLAGS = "0xF84"
RY_MALE, RY_FEMALE = 0.077, 0.016
RX_FEMALE, RX_MALE = 0.8, 0.6
AUTOSOMES = [str(i) for i in range(1, 23)]

COLUMNS = (["sample_name"] + [f"n_chr{c}" for c in AUTOSOMES] + ["n_chrX", "n_chrY",
           "r_y", "r_y_ci_low", "r_y_ci_high", "r_y_call",
           "r_x", "r_x_ci_low", "r_x_ci_high", "r_x_call", "sex_call"])


def count_reads(bam_path: str) -> tuple[dict[str, int], dict[str, int]]:
    """Filtered read counts and reference lengths for chr1-22, X, Y (keys without 'chr')."""
    import pysam
    with pysam.AlignmentFile(bam_path) as bam:
        names = dict(zip(bam.references, bam.lengths))
    prefix = "chr" if "chrX" in names else ""
    counts, lengths = {}, {}
    for c in AUTOSOMES + ["X", "Y"]:
        contig = prefix + c
        if contig not in names:
            raise SystemExit(f"{bam_path}: contig {contig} not in the BAM header")
        out = pysam.view("-c", "-q", MAPQ_MIN, "-F", EXCLUDE_FLAGS, bam_path, contig)
        counts[c], lengths[c] = int(out.strip()), names[contig]
    return counts, lengths


def sex_ratios(counts: dict[str, int], lengths: dict[str, int]) -> dict:
    n_x, n_y = counts["X"], counts["Y"]
    n = n_x + n_y
    r_y = n_y / n if n else float("nan")
    half = 1.96 * math.sqrt(r_y * (1 - r_y) / n) if n else float("nan")
    ry_lo, ry_hi = r_y - half, r_y + half
    ry_call = "male" if ry_lo > RY_MALE else "female" if ry_hi < RY_FEMALE else "unassigned"

    per_auto = [(n_x / lengths["X"]) / (counts[c] / lengths[c]) for c in AUTOSOMES if counts[c] > 0]
    k = len(per_auto)
    r_x = sum(per_auto) / k if k else float("nan")
    sd = math.sqrt(sum((v - r_x) ** 2 for v in per_auto) / (k - 1)) if k > 1 else float("nan")
    rx_lo, rx_hi = r_x - 1.96 * sd / math.sqrt(k), r_x + 1.96 * sd / math.sqrt(k)
    rx_call = "female" if rx_lo > RX_FEMALE else "male" if rx_hi < RX_MALE else "unassigned"

    sex = ry_call if ry_call == rx_call else "unassigned"
    return dict(r_y=r_y, r_y_ci_low=ry_lo, r_y_ci_high=ry_hi, r_y_call=ry_call,
                r_x=r_x, r_x_ci_low=rx_lo, r_x_ci_high=rx_hi, r_x_call=rx_call, sex_call=sex)


def summarize(per_sample_dir: str, out_csv: str) -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(per_sample_dir, "*.csv"))):
        if os.path.abspath(p) == os.path.abspath(out_csv):
            continue
        with open(p) as fh:
            lines = [l.strip() for l in fh.read().splitlines() if l.strip()]
        if len(lines) >= 2 and lines[0] == ",".join(COLUMNS):
            rows.append(lines[1])
    with open(out_csv, "w", newline="\n") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        fh.write("\n".join(rows) + "\n")
    calls = [r.split(",")[-1] for r in rows]
    tally = ", ".join(f"{c} {calls.count(c)}" for c in ("female", "male", "unassigned"))
    print(f"wrote {out_csv}: {len(rows)} samples; {tally}")


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    bam, out = sys.argv[1], sys.argv[2]
    counts, lengths = count_reads(bam)
    r = sex_ratios(counts, lengths)
    name = extract_sample_name(bam)
    vals = [name] + [str(counts[c]) for c in AUTOSOMES + ["X", "Y"]] + [
        f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in COLUMNS[25:]]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write(",".join(COLUMNS) + "\n")
        fh.write(",".join(vals) + "\n")
    print(f"{name}: R_Y {r['r_y']:.4f} [{r['r_y_ci_low']:.4f}, {r['r_y_ci_high']:.4f}] {r['r_y_call']}; "
          f"R_X {r['r_x']:.3f} [{r['r_x_ci_low']:.3f}, {r['r_x_ci_high']:.3f}] {r['r_x_call']} -> {r['sex_call']}")


if __name__ == "__main__":
    main()
