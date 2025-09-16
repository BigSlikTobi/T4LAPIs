#!/usr/bin/env python3
"""Quick functional check for the EmbeddingGenerator.

This script exercises the single and batch embedding pathways using the
SentenceTransformer fallback by default. It prints key diagnostics so you can
verify that embeddings are produced with the expected dimensions and metadata.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure the project's src/ directory is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nfl_news_pipeline.embedding.generator import EmbeddingGenerator
from nfl_news_pipeline.models import ContextSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an EmbeddingGenerator functionality check.")
    parser.add_argument(
        "--mode",
        choices=["openai", "transformer"],
        default="transformer",
        help="Primary embedding backend to exercise. Defaults to sentence-transformer.",
    )
    parser.add_argument(
        "--openai-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. Used only when --mode openai is selected.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size to use when instantiating the generator.",
    )
    parser.add_argument(
        "--sentence-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for fallback/primary local generation.",
    )
    return parser.parse_args()


def _format_vector_preview(vector: np.ndarray, limit: int = 8) -> str:
    preview = np.array2string(vector[:limit], precision=4, separator=", ")
    return preview.replace("\n", " ")


async def run_check(args: argparse.Namespace) -> None:
    use_openai = args.mode == "openai" and bool(args.openai_key)
    if args.mode == "openai" and not args.openai_key:
        logging.warning("--mode openai selected but no API key provided. Falling back to sentence-transformer.")
        use_openai = False

    generator = EmbeddingGenerator(
        openai_api_key=args.openai_key if use_openai else None,
        use_openai_primary=use_openai,
        batch_size=args.batch_size,
        sentence_transformer_model=args.sentence_model,
    )

    print("=== EmbeddingGenerator Functionality Check ===")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Backend mode: {'OpenAI' if use_openai else 'SentenceTransformer'}")

    model_info = generator.get_model_info()
    print("Model configuration:")
    for key, value in model_info.items():
        print(f"  - {key}: {value}")

    # Prepare sample summaries that mirror NFL news style content.
    summaries: List[ContextSummary] = [
        ContextSummary(
            news_url_id="test-story-001",
            summary_text="Patrick Mahomes leads Chiefs past Raiders with late touchdown drive.",
            llm_model="gpt-4o-mini",
            confidence_score=0.92,
            key_topics=["NFL", "Chiefs", "Quarterback"],
            entities={"players": ["Patrick Mahomes"], "teams": ["Kansas City Chiefs", "Las Vegas Raiders"]},
        ),
        ContextSummary(
            news_url_id="test-story-002",
            summary_text="Cowboys defense smothers Giants as Micah Parsons records three sacks.",
            llm_model="gpt-4o-mini",
            confidence_score=0.88,
            key_topics=["Defense", "Sacks", "Micah Parsons"],
            entities={"players": ["Micah Parsons"], "teams": ["Dallas Cowboys", "New York Giants"]},
        ),
    ]

    # Single embedding generation path.
    primary_summary = summaries[0]
    print("\nRequesting single embedding...")
    single_embedding = await generator.generate_embedding(primary_summary, primary_summary.news_url_id)
    single_embedding.validate()
    single_vector = np.array(single_embedding.embedding_vector, dtype=np.float32)
    print(
        "Single embedding result:"
        f"\n  - news_url_id: {single_embedding.news_url_id}"
        f"\n  - vector_length: {single_vector.size}"
        f"\n  - l2_norm: {np.linalg.norm(single_vector):.4f}"
        f"\n  - model_name: {single_embedding.model_name}"
        f"\n  - model_version: {single_embedding.model_version}"
        f"\n  - vector_preview: {_format_vector_preview(single_vector)}"
    )

    # Batch embedding generation path using both summaries.
    print("\nRequesting batch embeddings...")
    batch_embeddings = await generator.generate_embeddings_batch(summaries, [s.news_url_id for s in summaries])

    if not batch_embeddings:
        print("Batch embedding generation returned no results.")
        return

    print(f"Batch embedding result count: {len(batch_embeddings)}")
    for embedding in batch_embeddings:
        embedding.validate()
        vector = np.array(embedding.embedding_vector, dtype=np.float32)
        print(
            f"  - {embedding.news_url_id}: dim={vector.size}, l2_norm={np.linalg.norm(vector):.4f}, "
            f"preview={_format_vector_preview(vector)}"
        )

    print("\nAll embeddings validated successfully.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        asyncio.run(run_check(args))
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Embedding functionality check failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
