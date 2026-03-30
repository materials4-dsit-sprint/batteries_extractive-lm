This project automates the extraction of complex electrochemical properties from scientific literature by converting raw PDFs into a massive synthetic SQuAD-style dataset 
and fine-tuning domain-specific language models.

## The pipeline consists of three distinct phases:

### 1. Synthetic Generation

Raw PDFs processed using Docling to preserve table structures and chemical formulas in Markdown. We then leverage Claude 4.5 (Anthropic Batch API) to generate 
high-fidelity QA pairs.

Input: Research Papers (PDF)

Output: SQuAD 2.0 formatted JSON with is_impossible logic.

### 2. Ingestion & Processing 

To handle 140,000+ samples without memory crashe

Sliding Window: Context is processed with a 384-token window and a 128-token stride.

Schema Enforcement: Explicit Type-casting for character offsets to ensure F1/EM accuracy.

### 3. Fine-Tuning & Evaluation

Models are fine-tuned on either NVIDIA GPUs (CUDA) or Apple Silicon (MPS).

Metrics: F1, Exact Match (EM), BLEU, ROUGE-L, and BERTScore.


# Project Structure

<pre> 
├── data/
│   └── sample_squad.json       # A small snippet of QA pairs
├── scripts/
│   ├── extract_pdfs.py         # The Docling + Claude Parallel script
│   ├── build_dataset.py        # The Robust Generator/Flattening script
│   └── benchmark_models.py     # The evaluation script
├── results/
│
└── README.md      
</pre>

## References

1. Del Nostro, P.; Goldbeck, G.; Kienberger, F.; Moertelmaier, M.; Pozzi, A.; Al-Zubaidi-R-Smith, N.; Toti, D. Battery Testing Ontology: An EMMO-Based Semantic Framework for Representing Knowledge in Battery Testing and Battery Quality Control. Computers in Industry 2025, 164, 104203. https://doi.org/10.1016/j.compind.2024.104203.
2. Huang,S,  Cole, J.M.,  BatteryDataExtractor: battery-aware text-mining software embedded with BERT models - Chemical Science https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc04322j
3. Sayeed, H. M.; Clark, C.; Mohanty, T.; Sparks, T. KnowMat: An Agentic Approach to Transforming Unstructured Material Science Literature into Structured Data. November 3, 2025. https://doi.org/10.26434/chemrxiv-2025-l296q-v2.
4. Swain, M. C.; Cole, J. M. ChemDataExtractor: A Toolkit for Automated Extraction of Chemical Information from the Scientific Literature. J. Chem. Inf. Model. 2016, 56 (10), 1894–1904. https://doi.org/10.1021/acs.jcim.6b00207.
5. Zhang, T.; Kishore, V.; Wu, F.; Weinberger, K. Q.; Artzi, Y. BERTScore: Evaluating Text Generation with BERT. arXiv February 24, 2020. https://doi.org/10.48550/arXiv.1904.09675.
6. Parthasarathy, V. B.; Zafar, A.; Khan, A.; Shahid, A. The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities. arXiv August 23, 2024. https://doi.org/10.48550/arXiv.2408.13296.
7. De Baas, A.; Nostro, P. D.; Friis, J.; Ghedini, E.; Goldbeck, G.; Paponetti, I. M.; Pozzi, A.; Sarkar, A.; Yang, L.; Zaccarini, F. A.; Toti, D. Review and Alignment of Domain-Level Ontologies for Materials Science. IEEE Access 2023, 11, 120372–120401. https://doi.org/10.1109/ACCESS.2023.3327725.
8. Rajpurkar, P.; Zhang, J.; Lopyrev, K.; Liang, P. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing; Su, J., Duh, K., Carreras, X., Eds.; Association for Computational Linguistics: Austin, Texas, 2016; pp 2383–2392. https://doi.org/10.18653/v1/D16-1264.
