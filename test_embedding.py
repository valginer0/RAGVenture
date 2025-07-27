#!/usr/bin/env python3
"""Test script to check if sentence transformers work in Docker container."""

import sys
import traceback


def test_sentence_transformer():
    try:
        print("Testing sentence transformer loading...")
        from sentence_transformers import SentenceTransformer

        print("✓ SentenceTransformer import successful")

        print("Loading model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ Model loaded successfully!")

        print("Testing embedding generation...")
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text)
        print(f"✓ Embedding generated successfully! Shape: {embedding.shape}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False


def test_chroma():
    try:
        print("\nTesting Chroma vectorstore...")
        import langchain_chroma  # noqa: F401

        print("✓ Chroma import successful")
        return True

    except Exception as e:
        print(f"✗ Chroma error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Docker Embedding Test ===")

    st_success = test_sentence_transformer()
    chroma_success = test_chroma()

    if st_success and chroma_success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
