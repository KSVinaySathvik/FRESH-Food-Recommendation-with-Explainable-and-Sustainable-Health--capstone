# Data

Only small samples are committed. The full pipeline expects these files in `data/`:

| File | Produced by |
| --- | --- |
| `full_dataset.csv` | raw recipe dataset with an `NER` ingredient column |
| `claude_recipes_metadata_FINAL.json` | `pipeline/extract_metadata.py` + `pipeline/clean_recipes.py` |
| `claude_recipe_vectors_FINAL.npy` | `pipeline/generate_embeddings.py` |
| `claude_recipe_index.faiss`, `claude_recipe_ids_FINAL.npy` | `pipeline/build_faiss.py` |
| `new_chunks/` (`*_chunks.json`, `master_medical_index.index`) | `pipeline/ingest_nutrition.py` on the nutrition PDFs |
| `distilled_labels.csv` | `pipeline/distill_training_data.py` (committed) |
| `macro_vectors.pt` | macro-nutrient direction vectors used by `src/distilled_training.py` (not included) |
