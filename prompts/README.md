# Diagnostic QA generation prompts

Transcribed from Appendix D of the supplied camera-ready manuscript. PDF line breaks were normalized; `>=` represents the printed ≥. `{CONTEXT}` is the source-text placeholder. The output blocks describe a schema; alternatives separated by `|` are not literal valid JSON.

- `semantic_drift.txt` (D.1): 20 QA items per context, targeting 15 answerable and 5 unanswerable questions across seven probe families. The paper reports offline generation with DeepSeek-R1.
- `knowledge_overwriting_optional.txt` (D.2): an optional 12-item candidate-generation prompt. The released knowledge-overwriting benchmark retains **one question per context** from FaithEval and ConflictQA; it is not a 12-QA-per-context release.

D.3 retains difficult contexts: at 16× compression, remove a candidate/context only if every Qwen3 compressor (0.6B, 1.7B, 4B, 8B, 32B) exceeds 0.8 normalized EM; for semantic drift, average EM over the context's 20 QA items first. Generation constraints describe the intended protocol; the release audit records deviations present in the submitted data.
