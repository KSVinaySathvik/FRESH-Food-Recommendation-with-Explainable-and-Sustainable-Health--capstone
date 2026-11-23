"""Rebuild the retrieval artifacts: metadata extraction, embeddings, FAISS index.

    python pipeline/run_pipeline.py
"""

import os
import sys
import time
from datetime import datetime

def print_step(step_num, title):
    """Print formatted step header"""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70 + "\n")

def run_pipeline():
    """Execute complete pipeline:
    1. Extract metadata from CSV
    2. Generate embeddings
    3. Build faiss index
    """

    start_time = time.time()

    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "RECIPE RECOMMENDATION PIPELINE" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # STEP 1: Extract Metadata
        print_step(1, "Extracting Metadata from CSV")

        from extract_metadata import extract_metadata_from_csv

        recipes = extract_metadata_from_csv(
            csv_path='data/full_dataset.csv',
            output_path='data/claude_recipes_metadata_FINAL.json'
        )

        if not recipes or len(recipes) == 0:
            print("ERROR: No recipes extracted!")
            return False

        print(f"✓ Step 1 completed: {len(recipes):,} recipes extracted")

        # STEP 2: Generate Embeddings
        print_step(2, "Generating Recipe Embeddings")

        from generate_embeddings import generate_recipe_embeddings

        embeddings = generate_recipe_embeddings(
            metadata_path='data/claude_recipes_metadata_FINAL.json',
            output_path='data/claude_recipe_vectors_FINAL.npy',
            batch_size=64
        )

        if embeddings is None or len(embeddings) == 0:
            print("ERROR: No embeddings generated!")
            return False

        print(f"✓ Step 2 completed: {len(embeddings):,} vectors generated")

        # STEP 3: Build FAISS Index
        print_step(3, "Building FAISS Index")

        from build_faiss import build_faiss_index

        index, recipe_ids = build_faiss_index(
            vectors_path='data/claude_recipe_vectors_FINAL.npy',
            metadata_path='data/claude_recipes_metadata_FINAL.json',
            index_path='data/claude_recipe_index.faiss',
            ids_path='data/claude_recipe_ids_FINAL.npy',
            use_gpu=False
        )

        if index is None:
            print("ERROR: Failed to build FAISS index!")
            return False

        print(f"✓ Step 3 completed: Index with {index.ntotal:,} vectors")

        # Pipeline Complete
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nGenerated files:")
        print(f"  ✓ recipes_metadata_FINAL.json    - {len(recipes):,} recipes")
        print(f"  ✓ recipe_vectors_FINAL.npy        - {len(embeddings):,} vectors")
        print(f"  ✓ recipe_index.faiss              - {index.ntotal:,} indexed")
        print(f"  ✓ recipe_ids_FINAL.npy            - {len(recipe_ids):,} IDs")

        print(f"\nTotal time: {hours}h {minutes}m {seconds}s")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"\nPIPELINE FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
