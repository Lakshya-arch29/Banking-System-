from pathlib import Path
import sys

from app.services.matching.embeddings import (
    EMBEDDING_DIM,
    generate_embedding,
    get_embedding_model,
    get_embedding_model_name,
    resolve_model_name,
)


TEST_TEXT = "HEX BOLT M8x25 SS304"


def main() -> int:
    print("=" * 72)
    print("MIRA EMBEDDING MODEL VERIFICATION")
    print("=" * 72)
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print()

    resolved_model = resolve_model_name()
    public_model_name = get_embedding_model_name()

    print(f"Resolved model: {resolved_model}")
    print(f"Reported model name: {public_model_name}")
    print(f"Expected dimension: {EMBEDDING_DIM}")
    print()

    if "AshIndian" not in resolved_model and "Mira.ai" not in resolved_model:
        print("MODEL SELECTION: FAILED")
        print(
            "The application is not selecting AshIndian/Mira.ai. "
            "It is selecting another model."
        )
        return 1

    print("MODEL SELECTION: PASSED")
    print()
    print("Loading model...")

    try:
        model = get_embedding_model()
    except Exception as exc:
        print("MODEL LOAD: FAILED")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    print("MODEL LOAD: PASSED")
    print(f"Loaded model class: {type(model).__name__}")
    print()
    print("Generating test embedding...")

    try:
        embedding = generate_embedding(TEST_TEXT)
    except Exception as exc:
        print("EMBEDDING GENERATION: FAILED")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    actual_dimension = len(embedding)

    print("EMBEDDING GENERATION: PASSED")
    print(f"Test text: {TEST_TEXT}")
    print(f"Embedding length: {actual_dimension}")
    print()

    if actual_dimension != EMBEDDING_DIM:
        print("DIMENSION CHECK: FAILED")
        print(
            f"Expected {EMBEDDING_DIM}, "
            f"but received {actual_dimension}."
        )
        return 1

    print("DIMENSION CHECK: PASSED")
    print()
    print("FINAL RESULT: AshIndian/Mira.ai is loaded and in use.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())