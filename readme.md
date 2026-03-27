run instructions:

python caption_images.py \
  --input-jsonl americasnlp2026/data/pilot/guarani.jsonl \
  --base-path americasnlp2026/data/pilot \
  --language guarani \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --output-jsonl outputs/guarani_pilot_captioned.jsonl \
  --output-txt outputs/guarani_pilot_spanish.txt
  
  
  
  python translate_mt.py train \
  --language guarani \
  --model-name facebook/mbart-large-50 \
  --train-src data/combined_train.es \
  --train-tgt data/combined_train.gn \
  --val-src data/combined_val.es \
  --val-tgt data/combined_val.gn \
  --output-dir runs/mbart_es_gn \
  --fp16
  
  
  python translate_mt.py predict \
  --language guarani \
  --model-path runs/mbart_es_gn \
  --input-file outputs/guarani_pilot_spanish.txt \
  --output-file outputs/guarani_pilot_pred.txt
  
  
  python score_outputs.py \
  --pred-file outputs/guarani_pilot_pred.txt \
  --ref-file data/guarani_refs.txt \
  --metrics-json outputs/guarani_metrics.json \
  --per-example-tsv outputs/guarani_scores.tsv
  
  python score_outputs.py \
  --pred-file outputs/guarani_pilot_pred.txt \
  --ref-file americasnlp2026/data/pilot/guarani.jsonl \
  --ref-format jsonl \
  --ref-field target_caption \
  --metrics-json outputs/guarani_metrics.json \
  --per-example-tsv outputs/guarani_scores.tsv
  