# Team Gators — AmericasNLP 2026 — V1 Submission Configs

## Pipeline
Image → Qwen2.5-VL-[7B|72B]-Instruct (4-bit) → Spanish caption → Gemini 2.5 Flash (BM25 many-shot) → Target language

## Per-language configs

| Language | VLM model | r | d | temp | Notes |
|----------|-----------|---|---|------|-------|
| Guaraní  | 72B (v1) / 7B (v0) | 80 | 49 | 0.0 | v1 fixes 8B→72B caption mismatch |
| Maya     | 8B | 0 | 49 | 0.0 | No retrieval — dummy corpus |
| Nahuatl  | 8B | 40 | 20 | 0.0 | |
| Wixarika | 8B | 40 | 20 | 0.0 | |
| Bribri   | 8B | 80 | 20 | 0.0 | v1: NFD normalization + stronger prompt |

## Key v0→v1 changes

### Guaraní
- v0: test captions from Qwen2.5-VL-7B-Instruct
- v1: test captions from Qwen2.5-VL-72B-Instruct (4-bit, same model used for dev eval)
- Dev chrF++ improvement: 41.48 → 48.24

### Bribri
- v0: NFC-encoded predictions scored against NFD references (silent penalty on chrF++)
- v1: predictions NFD-normalized to match reference encoding
- v1: system prompt explicitly describes Bribri SOV order, tonal diacritics, consonant clusters
- Dev chrF++ improvement (NFD-normalized): 17.02 → 19.99

## Reproducibility notes
- BM25 retrieval: rank_bm25, BM25Okapi, default params
- Dev exemplars: leave-one-out by exact string match on normalized lowercase Spanish
- All translation at temperature=0.0 except targeted retranslation of degenerate Bribri predictions (temp=0.3 and 0.5)
- Bribri post-processing: punctuation-normalized dedup + targeted retranslation of 15 degenerate predictions
- Unicode: all Bribri predictions normalized to NFD before submission (references are NFD; all other language references are NFC)
