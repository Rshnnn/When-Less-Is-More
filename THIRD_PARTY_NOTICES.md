# Third-party resources and attribution

This release reorganizes the diagnostic data supplied with the paper. Original JSON entries are archived without modification; JSONL files add stable identifiers, descriptive field names, and audit metadata. No upstream training code is copied into this repository.

| Resource | Role | Source and terms |
|:--|:--|:--|
| PCC (Dai et al., ACL 2025) | Compressor–decoder paradigm and upstream implementation | [Paper](https://aclanthology.org/2025.acl-long.1394/) · [Official code](https://github.com/microsoft/AnthropomorphicIntelligence/tree/main/PCC). Follow the upstream repository's license when reusing its code. |
| FaithEval (Ming et al.) | Filtered counterfactual contexts and QA; contexts for semantic-drift probes | [Dataset](https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0) · [Repository](https://github.com/SalesforceAIResearch/FaithEval). The repository declares Apache-2.0; underlying source-dataset terms also apply. |
| ConflictQA (Xie et al., ICLR 2024) | StrategyQA/GPT-4 knowledge-conflict examples | [Repository](https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict), Apache-2.0; derived from StrategyQA and subject to applicable source terms. |
| FineWeb (Penedo et al., NeurIPS 2024) | Source passages for semantic-drift probes | [Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb), ODC-By 1.0; use is also subject to Common Crawl's terms. Underlying web content retains its original rights. |

The source data supplied for packaging did not include per-passage FineWeb URLs, crawl identifiers, the exact upstream dataset revisions, or an explicit license for the new QA annotations. These identifiers and rights are not invented here. Original rights and notices remain applicable; this document does not grant a blanket license to third-party content. Consult upstream sources and the authors for reuse terms beyond those already granted upstream.

Paper figures and prompt text are reproduced from the authors' supplied camera-ready manuscript. Citation metadata follows that manuscript; no proceedings volume or page numbers have been invented.
