This project automates the extraction of complex electrochemical properties from scientific literature by converting raw PDFs into a massive synthetic SQuAD-style dataset 
and fine-tuning domain-specific language models.

The pipeline consists of three distinct phases:

1. Ingestion & Synthetic Generation

Raw PDFs processed using Docling to preserve table structures and chemical formulas in Markdown. We then leverage Claude 4.5 (Anthropic Batch API) to generate 
high-fidelity QA pairs.

Input: Research Papers (PDF)

Output: SQuAD 2.0 formatted JSON with is_impossible logic.

2. Robust Data Engineering

To handle 140,000+ samples without memory crashe

Sliding Window: Context is processed with a 384-token window and a 128-token stride.

Schema Enforcement: Explicit Type-casting for character offsets to ensure F1/EM accuracy.

3. Fine-Tuning & Evaluation

Models are fine-tuned on either NVIDIA GPUs (CUDA) or Apple Silicon (MPS).

Metrics: F1, Exact Match (EM), BLEU, ROUGE-L, and BERTScore.


Directory Tree

├── data/
│   └── sample_squad.json       # A small snippet of QA pairs
├── scripts/
│   ├── extract_pdfs.py         # The Docling + Claude Parallel script
│   ├── build_dataset.py        # The Robust Generator/Flattening script
│   └── benchmark_models.py     # The evaluation script
├── results/
│
└── README.md                   
