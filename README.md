# GNN for Key-Value Pair Linking in Documents

Experimental scripts for extracting key-value pairs from digitized documents using a lightweight GNN with selective edge construction. The repository also includes supporting components for semantic normalization, table processing, and pipeline validation used around the main KVP-linking experiments.

**Authors:** Gabriel Sichelero and Ricardo Dutra da Silva.

![Technical pipeline](assets/pipeline_architecture_paper.png)

## Objective

This repository collects the main implementation and experiment scripts, with the primary focus on GNN-based key-value pair linking on public datasets such as FUNSD and WildReceipt. Code is kept separate from datasets, trained checkpoints, and large experiment outputs.

Datasets, model checkpoints, large generated outputs, and private documents are not included. Images under `assets/` are explanatory figures or anonymized examples used to contextualize the repository.

## Overview

![Anonymized KVP annotation example](assets/invoice_kvp_annotations_anon.png)

The core contribution is a GNN model that represents each document as a graph: text blocks are nodes, and candidate key-value relations are edges. Selective edge construction removes semantically unlikely pairs and focuses training on `question/header -> answer/other` links.

In the full document-processing pipeline, the GNN is the central KVP-linking stage:

1. region segmentation;
2. OCR to obtain text and coordinates;
3. key-value pair linking with the proposed GNN;
4. table structure recognition;
5. semantic normalization into standardized categories;
6. structured JSON output.

## Main Scripts

| File | Purpose |
| --- | --- |
| `scripts/kvp_gnn_cross_dataset.py` | GNN experiments for key-value pair linking, including WildReceipt -> FUNSD conversion, FUNSD+WildReceipt combined training, and cross-dataset evaluation. |
| `scripts/kv_extractor_ml.py` | Support module for extracting and organizing key-value pairs used by the pipeline. |
| `scripts/semantic_normalization_suite.py` | Tests for value normalization, including dates, monetary values, numbers, and semantic classification. |
| `scripts/semantic_normalization_gt_context.py` | Trains semantic embeddings with ground-truth context from KVP pairs and table headers/rows. |
| `scripts/invoice_categories.py` | Lightweight vocabulary of invoice semantic categories used by the normalization scripts. |
| `scripts/unified_document_pipeline.py` | Integrated OCR, table detection, KVP extraction, classification, and result-export pipeline. |
| `scripts/test_full_pipeline.py` | End-to-end validation script for document images, with OCR, table, KVP, and category visualizations. |
| `scripts/run_full_pipeline_with_semantic_context.py` | Runner for semantic-context training plus optional evaluation and visualization steps. |
| `scripts/table_structure_detector.py` | Table detection and table-structure recognition module. |
| `scripts/locator_classifier.py` | Auxiliary classifier used by the integrated categorization stage. |

## Reference Results

GNN results on public KVP-linking benchmarks:

| Setting | Reported result |
| --- | --- |
| GNN trained and tested on FUNSD | F1 = 0.772 with approximately 890K parameters |
| GNN trained on FUNSD+WildReceipt and tested on FUNSD | F1 = 0.832 |
| GNN trained on FUNSD+WildReceipt and tested on WildReceipt | F1 = 0.721 |

![Public KVP evaluation protocol](assets/kvp_public_protocol_paper.png)

## How to Run

Install the main dependencies:

```bash
pip install -r requirements.txt
```

Example commands from the repository root:

```bash
python scripts/kvp_gnn_cross_dataset.py
python scripts/semantic_normalization_suite.py
python scripts/test_full_pipeline.py --image path/to/document.png --output pipeline_results
```

By default, scripts assume they are being run from this repository root. To point them to external data or a different local layout, define:

```bash
set DOCUMENT_KVP_PROJECT_ROOT=C:\path\to\repository-or-data
set KVP_DATASETS_DIR=C:\path\to\kvp_datasets
```

On Linux/macOS, use `export` instead of `set`.

## Expected Data

Full experiment runs may require external data such as:

- FUNSD;
- WildReceipt;
- document images and annotations used in local pipeline experiments;
- previously trained checkpoints when running evaluation-only modes.

These files are not redistributed here. This repository is intentionally limited to source code, documentation, and lightweight figures.

## Structure

```text
.
├── README.md
├── requirements.txt
├── assets/
│   ├── pipeline_architecture_paper.png
│   ├── invoice_kvp_annotations_anon.png
│   └── kvp_public_protocol_paper.png
└── scripts/
    ├── kvp_gnn_cross_dataset.py
    ├── semantic_normalization_suite.py
    ├── semantic_normalization_gt_context.py
    ├── invoice_categories.py
    ├── unified_document_pipeline.py
    ├── test_full_pipeline.py
    ├── run_full_pipeline_with_semantic_context.py
    ├── table_structure_detector.py
    ├── kv_extractor_ml.py
    └── locator_classifier.py
```
