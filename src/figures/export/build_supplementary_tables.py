"""Supplementary Tables S1, S6 and S7 as submission-ready Excel files (no_deep cohort), plus their LaTeX table entries.

    S1  Participants of the Cristiano et al. cohort        <- data/no_deep/tableS_participants.csv
    S6  Classes and enrichment of the C8 signatures          <- data/no_deep/tableS_c8_classes.csv
    S7  Configuration and counts per outer fold             <- data/no_deep/tableS_folds_counts_{A_per_fold,B_pooled}.csv
(tables S2-S5 are printed in the supplement PDF)

Run from the repository root, after the exporters that write the source CSVs:
    COHORT_VARIANT=no_deep python src/figures/export/export_tableS_folds_counts.py     # S1, S7
    COHORT_VARIANT=no_deep python src/figures/export/export_tableS_c8_classes.py       # S6
    python src/figures/export/build_supplementary_tables.py

Writes submission/supplementary_tables/ (git-ignored, regenerated from the CSVs):
    Supplementary_Table_S1.xlsx, _S6.xlsx, _S7.xlsx   README sheet (title, legend, column definitions) + data sheet(s)
    supplementary_tables.tex                            the table entries for the supplement (caption + label)
    README.txt                                          what the folder holds and how to rebuild it
Every column must have a definition; row counts are asserted.
"""
import datetime
import os
import re
import sys

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, REPO_ROOT  # noqa: E402

OUT = os.path.join(REPO_ROOT, "submission", "supplementary_tables")
TASK_NAMES = {"brca_crc": "BRCA/CRC versus healthy", "pancan": "pan-cancer versus healthy"}
C8_CLASS_NAMES = {"blood / immune": "blood and immune cells", "epithelial / parenchymal": "epithelial and parenchymal cells",
                  "neural / glial": "neural and glial cells", "stromal / mesenchymal": "stromal and mesenchymal cells",
                  "endothelial": "endothelial cells", "other": "other"}


def per_task(defs):
    """Expand '<task>' in column names into both tasks."""
    out = []
    for name, text in defs:
        if "<task>" in name:
            for t, tn in TASK_NAMES.items():
                out.append((name.replace("<task>", t), text.replace("<task name>", tn)))
        else:
            out.append((name, text))
    return out


TABLES = {
    "S1": dict(
        label="tab:sup_participants",
        title="Participants of the Cristiano et al. cohort",
        legend=("One row per FinaleDB sequencing record (534 records of 473 individuals): role in this study, reason for any "
                "exclusion, task membership and, per task, the outer cross-validation fold, the out-of-fold probability of the "
                "data-driven model with its 95%-specificity threshold and call, the out-of-fold scores of the ATAC-seq and RNA-seq "
                "baselines, and the ichorCNA tumor fraction. Task columns are empty for records outside the task."),
        sheets=[("Participants", "tableS_participants.csv")],
        columns=per_task([
            ("ID", "FinaleDB record identifier."),
            ("sample_name", "Sample name in Cristiano et al. (2019); second sequencing runs of healthy participants carry the suffix _Rep."),
            ("participant_id", "Sample name without _Rep. Serial lung cancer samples keep their time-point suffix, so the 534 records "
                               "have 517 identifiers for 473 individuals."),
            ("disease", "FinaleDB label. 'Lung cancer naive nsclc' = treatment-naive non-small cell lung cancer, the lung group of the "
                        "pan-cancer task; 'Lung cancer' records are excluded (see exclusion_reason)."),
            ("material", "plasma_cfDNA, or lymphocyte_gDNA for the two lymphocyte genomic DNA samples (CGH14, CGH15)."),
            ("role", "panel_of_normals: one of the 50 healthy samples used only for the TSS quality filter and the ichorCNA panel of "
                     "normals; model: enters the classification tasks; excluded: not used."),
            ("exclusion_reason", "replicate_second_run: second sequencing run of a participant (replicate_of gives the first); "
                                 "serial_monitoring: lung cancer samples taken before and during treatment; "
                                 "lung_not_labelled_naive_nsclc: single lung cancer samples that FinaleDB does not label as "
                                 "treatment-naive NSCLC; lymphocyte_gDNA: not plasma cfDNA; single_case: the one duodenal cancer; "
                                 "high_coverage_substudy: the 30 healthy participants of the high-coverage sub-study of "
                                 "Cristiano et al. (sequenced at 6-14x)."),
            ("replicate_second_run", "TRUE for a second sequencing run of a participant."),
            ("replicate_of", "For a second run, the ID of the participant's first run."),
            ("in_task_brca_crc", "TRUE if the record is in the BRCA/CRC versus healthy task (244 records)."),
            ("in_task_pancan", "TRUE if the record is in the pan-cancer versus healthy task (367 records)."),
            ("n_fragments", "Number of fragments in the FinaleDB fragment file."),
            ("fold_<task>", "<task name>: outer cross-validation fold (1-10) in which the record was a test sample."),
            ("oof_probability_<task>", "<task name>: out-of-fold probability of cancer from the data-driven model trained without "
                                       "the record's fold."),
            ("threshold_95spec_<task>", "<task name>: threshold of that fold, the 95%-specificity point of its training partition."),
            ("called_cancer_<task>", "<task name>: TRUE if the out-of-fold probability is at or above the threshold."),
            ("oof_score_atac_<task>", "<task name>: out-of-fold score of the ATAC-seq baseline, from the model of the same outer fold."),
            ("oof_score_rna_<task>", "<task name>: out-of-fold score of the RNA-seq baseline, from the model of the same outer fold."),
            ("ichorcna_tumour_fraction", "ichorCNA tumor fraction, the score of the ichorCNA baseline (records in a task)."),
        ]),
        formats={"n_fragments": "#,##0", "fold_": "0", "oof_": "0.0000", "threshold_": "0.0000", "ichorcna_": "0.0000"},
        checks=lambda s: (len(s["Participants"]) == 534 and s["Participants"]["ID"].is_unique
                          and int(s["Participants"]["in_task_brca_crc"].sum()) == 244
                          and int(s["Participants"]["in_task_pancan"].sum()) == 367),
    ),
    "S6": dict(
        label="tab:sup_c8_classes",
        title="Classes and enrichment of the MSigDB C8 cell-type signatures",
        legend=("For each of the 738 signatures tested (15 to 500 genes among the 14,221 TSSs): the class of cell type assigned "
                "from its name with fixed keyword lists, without reference to the results, the number of TSSs, and per task the "
                "enrichment z, the q value and the direction against random sets of TSSs with the same GC content, coverage and "
                "dip depth in healthy plasma."),
        sheets=[("C8 signatures", "tableS_c8_classes.csv")],
        columns=per_task([
            ("signature", "MSigDB C8 cell-type signature (version 2024.1)."),
            ("study_tissue_prefix", "Study and tissue part of the name, removed before the class is assigned."),
            ("cell_type_part", "Cell-type part of the name, from which the class is read."),
            ("class_of_cell_type", "Class assigned with fixed keyword lists on the name only, without reference to the results; "
                                   "signatures from bone marrow and cord blood that match no list count as blood and immune cells; "
                                   "'other' = no cell type in the name (for example clusters named after marker genes)."),
            ("n_tss", "Number of TSSs of the signature among the 14,221."),
            ("z_<task>", "<task name>: enrichment, the mean rank of the signature's TSSs along the ranking by Cohen's d relative "
                         "to 10,000 random sets of TSSs with the same composition over quartiles of GC content, coverage and dip "
                         "depth in the panel of normals, in units of their standard deviation; negative = less active in plasma of "
                         "patients with cancer."),
            ("q_<task>", "<task name>: two-sided empirical p value among the random sets, Benjamini-Hochberg corrected over the "
                         "738 signatures."),
            ("direction_<task>", "<task name>: for q < 0.05, less or more active in cancer."),
        ]),
        formats={"n_tss": "0", "z_": "0.00", "q_": "0.0000"},
        checks=lambda s: len(s["C8 signatures"]) == 738 and s["C8 signatures"]["signature"].is_unique,
    ),
    "S7": dict(
        label="tab:sup_folds",
        title="Selected configuration and confusion counts per outer fold",
        legend=("Part A: per task, method and outer fold, the configuration selected in the inner cross-validation, the inner and "
                "outer AUROC, the 95%-specificity threshold set on the training partition, and the counts on the test fold. "
                "Part B: the counts pooled over the ten test folds with exact binomial (Clopper-Pearson) 95% confidence intervals "
                "and the range of the per-fold specificity, and the Jiang et al. cohort scored at the threshold fixed on the "
                "Cristiano et al. cohort, for HCC against all non-HCC participants and against each comparison group."),
        sheets=[("A per outer fold", "tableS_folds_counts_A_per_fold.csv"), ("B pooled", "tableS_folds_counts_B_pooled.csv")],
        columns=[
            ("task", "Classification task."),
            ("method", "Data-driven (the TSS model), ATAC-seq and RNA-seq (hypothesis-driven region sets), ichorCNA (tumor fraction)."),
            ("fold", "A: outer cross-validation fold."),
            ("feature", "A: fragmentomic feature selected in the inner cross-validation (ATAC-seq: TFBS features; RNA-seq: RCOV)."),
            ("ranking", "A: ranking function selected in the inner cross-validation ('none' where nothing is ranked)."),
            ("k", "A: number of regions kept by the ranking ('all' = all 14,221 TSSs, or all regions of a baseline)."),
            ("classifier", "A: classifier selected in the inner cross-validation."),
            ("hyperparameters", "A: its hyperparameters selected in the inner cross-validation."),
            ("inner_cv_auroc", "A: mean inner cross-validation AUROC of the selected configuration."),
            ("test_auroc", "A: AUROC on the outer test fold."),
            ("threshold_95spec", "A: score threshold at 95% specificity on the training partition of the fold."),
            ("TP", "True positives (cancers called cancer) at that threshold."),
            ("FN", "False negatives (cancers called healthy)."),
            ("TN", "True negatives (healthy or non-cancer participants called healthy)."),
            ("FP", "False positives (healthy or non-cancer participants called cancer)."),
            ("sensitivity", "B: pooled sensitivity, TP / (TP + FN)."),
            ("sensitivity_lo", "B: lower bound of its exact binomial 95% confidence interval."),
            ("sensitivity_hi", "B: upper bound of its exact binomial 95% confidence interval."),
            ("specificity", "B: pooled specificity, TN / (TN + FP)."),
            ("specificity_lo", "B: lower bound of its exact binomial 95% confidence interval."),
            ("specificity_hi", "B: upper bound of its exact binomial 95% confidence interval."),
            ("specificity_fold_min", "B: lowest specificity of a single outer test fold."),
            ("specificity_fold_max", "B: highest specificity of a single outer test fold."),
        ],
        formats={"fold": "0", "inner_cv_auroc": "0.000", "test_auroc": "0.000", "threshold_95spec": "0.0000",
                 "TP": "0", "FN": "0", "TN": "0", "FP": "0", "sensitivity": "0.000", "sensitivity_": "0.000",
                 "specificity": "0.000", "specificity_": "0.000"},
        checks=lambda s: len(s["A per outer fold"]) == 80 and len(s["B pooled"]) == 20,
    ),
}

HEADER_FILL = PatternFill("solid", fgColor="DDE4EE")
THIN = Side(style="thin", color="8C96A0")


def number_format(col, formats):
    for key, fmt in formats.items():
        if col == key or (key.endswith("_") and col.startswith(key)):
            return fmt
    return None


def write_table(no, spec, frames, sources):
    path = os.path.join(OUT, f"Supplementary_Table_{no}.xlsx")
    defs = dict(spec["columns"])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        # README sheet
        rows = [(c, defs[c]) for c in dict.fromkeys(c for f in frames.values() for c in f.columns)]
        pd.DataFrame(rows, columns=["Column", "Definition"]).to_excel(xw, sheet_name="README", index=False, startrow=5)
        ws = xw.sheets["README"]
        ws["A1"] = f"Supplementary Table {no}. {spec['title']}."
        ws["A1"].font = Font(bold=True, size=13)
        ws["A2"] = spec["legend"]
        ws.merge_cells("A2:B2")
        ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[2].height = 15 * (len(spec["legend"]) // 120 + 2)
        sheets = ", ".join(f"'{n}'" for n in frames)
        ws["A4"] = (f"Data sheet{'s' if len(frames) > 1 else ''}: {sheets}. Source: {', '.join(sources)} (src/figures/data/no_deep), "
                    f"built by src/figures/export/build_supplementary_tables.py on {datetime.date.today().isoformat()}.")
        ws["A4"].font = Font(italic=True, size=9, color="555555")
        ws.merge_cells("A4:B4")
        for cell in ws[6]:
            cell.font = Font(bold=True)
            cell.fill = HEADER_FILL
            cell.border = Border(bottom=THIN)
        for r in ws.iter_rows(min_row=7):
            r[0].font = Font(name="Consolas", size=10)
            for cell in r:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 120
        ws.sheet_view.showGridLines = False
        # data sheets
        for name, df in frames.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for j, col in enumerate(df.columns, 1):
                letter = get_column_letter(j)
                head = ws.cell(1, j)
                head.font = Font(bold=True)
                head.fill = HEADER_FILL
                head.border = Border(bottom=THIN)
                head.alignment = Alignment(vertical="center")
                fmt = number_format(col, spec["formats"])
                if fmt:
                    for cell in ws[letter][1:]:
                        cell.number_format = fmt
                sample = [len(str(v)) for v in df[col].head(300) if pd.notna(v)]
                width = max([len(col)] + [min(s, 60) for s in sample]) + 2
                ws.column_dimensions[letter].width = min(width, 62)
    return path


def tex_entries():
    out = ["% Supplementary Tables S1, S6 and S7, provided as separate Excel files.",
           "% Written by src/figures/export/build_supplementary_tables.py (same titles and legends as the README sheets).", ""]
    esc = lambda s: s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("et al.", r"\textit{et al.}")
    for no, spec in TABLES.items():
        out += [r"\begin{table}[H]",
                rf"\caption{{\textbf{{{esc(spec['title'])}}} (separate Excel file, \texttt{{Supplementary\_Table\_{no}.xlsx}}). "
                rf"{esc(spec['legend'])} A README sheet defines every column.}}",
                rf"\label{{{spec['label']}}}", r"\end{table}", ""]
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    for no, spec in TABLES.items():
        frames, sources = {}, []
        for sheet, csv in spec["sheets"]:
            df = pd.read_csv(os.path.join(DATA_DIR, csv), keep_default_na=True)
            if spec["label"] == "tab:sup_folds":    # method names as in the manuscript
                df["method"] = df["method"].replace({"ATAC": "ATAC-seq", "RNA": "RNA-seq"})
            if spec["label"] == "tab:sup_c8_classes":
                df["class_of_cell_type"] = df["class_of_cell_type"].map(C8_CLASS_NAMES)
                assert df["class_of_cell_type"].notna().all(), "unknown C8 class"
            frames[sheet] = df
            sources.append(csv)
        missing = [c for f in frames.values() for c in f.columns if c not in dict(spec["columns"])]
        assert not missing, f"{no}: columns without a definition: {missing}"
        assert spec["checks"](frames), f"{no}: row counts differ from the manuscript"
        written.append(write_table(no, spec, frames, sources))
        print(f"{no}: " + ", ".join(f"{s} {len(f)} rows x {f.shape[1]} columns" for s, f in frames.items()))
    with open(os.path.join(OUT, "supplementary_tables.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tex_entries())
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("Supplementary Tables S1, S6 and S7 of the revised manuscript, as uploaded with the submission.\n\n"
                 + "\n".join(f"  {os.path.basename(p)}" for p in written)
                 + "\n  supplementary_tables.tex  the table entries (caption + label) to paste into the supplement\n\n"
                 "Rebuild (repository root):\n"
                 "  COHORT_VARIANT=no_deep python src/figures/export/export_tableS_folds_counts.py\n"
                 "  COHORT_VARIANT=no_deep python src/figures/export/export_tableS_c8_classes.py\n"
                 "  python src/figures/export/build_supplementary_tables.py\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
