# External annotations (not in the repository: download each file into this folder)

| File | Source | Version | URL | Downloaded |
|---|---|---|---|---|
| `msigdb/h.all.v2024.1.Hs.symbols.gmt` | MSigDB Hallmark collection (H), gene symbols | 2024.1.Hs | https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt | 28 Oct 2026 |
| `msigdb/c8.all.v2024.1.Hs.symbols.gmt` | MSigDB cell-type signature collection (C8), gene symbols | 2024.1.Hs | https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c8.all.v2024.1.Hs.symbols.gmt | 28 Oct 2026 |
| `msigdb/c2.cp.reactome.v2024.1.Hs.symbols.gmt` | MSigDB Reactome pathways (C2:CP:REACTOME), gene symbols | 2024.1.Hs | https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.reactome.v2024.1.Hs.symbols.gmt | |
| `hpa/proteinatlas.tsv` (+ .zip) | Human Protein Atlas consensus data (RNA tissue / blood cell / blood lineage specificity) | 25.1 (file dated 3 Jun 2026) | https://www.proteinatlas.org/download/proteinatlas.tsv.zip | 3 Nov 2026 |
| `GRCh38_centromere_acen.txt` | Centromere (acen) bands, GRCh38, from the ichorCNA package extdata | ichorCNA | https://github.com/broadinstitute/ichorCNA | 2 Nov 2026 |

Used by `src/figures/export/export_figS18_gene_sets.py` (MSigDB Hallmark and C8, HPA), `export_figS18_matched_10k.py` and `export_figS18_reactome_healthy.py` (Reactome) and `src/comp/build_tss_set_matrices.py` (centromeres).
