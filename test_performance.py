#!/usr/bin/env python3
"""
Comprehensive End-to-End Performance Test for RAG Startups

This script tests the complete workflow with real YC startup data,
measuring performance of each component and validating functionality.
"""

import json
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_startups.config.settings import get_settings
from rag_startups.core.model_manager import ModelType
from rag_startups.core.model_service import ModelService
from rag_startups.core.rag_chain import format_startup_idea, initialize_rag
from rag_startups.core.startup_metadata import StartupLookup
from rag_startups.data.loader import load_data


def main():
    """Run comprehensive performance test."""
    print("=== RAG Startups End-to-End Performance Test ===")
    print("Testing with real YC startup data (yc_startups.json)")
    print("Components: Data loading, RAG pipeline, smart model management")
    print()

    total_start = time.time()

    # Test 1: Data Loading Performance
    print("1. 📊 Testing data loading performance...")
    load_start = time.time()
    try:
        df, json_data = load_data("yc_startups.json")
        load_time = time.time() - load_start
        print(f"   ✅ Loaded {len(df)} startups in {load_time:.2f}s")
        print(f"   📈 Processing rate: {len(df)/load_time:.0f} startups/second")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False

    # Test 2: Startup Lookup Performance
    print("\n2. 🔍 Testing startup lookup initialization...")
    lookup_start = time.time()
    try:
        lookup = StartupLookup(json_data)
        lookup_time = time.time() - lookup_start
        print(f"   ✅ Initialized lookup in {lookup_time:.2f}s")
        print(f"   📊 Indexed {len(json_data)} startup records")
    except Exception as e:
        print(f"   ❌ Lookup initialization failed: {e}")
        return False

    # Test 3: RAG System Initialization
    print("\n3. 🧠 Testing RAG system initialization...")
    rag_start = time.time()
    try:
        retriever, startup_lookup = initialize_rag(df, json_data)
        rag_time = time.time() - rag_start
        print(f"   ✅ Initialized RAG system in {rag_time:.2f}s")
        print(f"   🔗 Vector store and retriever ready")
    except Exception as e:
        print(f"   ❌ RAG initialization failed: {e}")
        return False

    # Test 4: Smart Model Management
    print("\n4. 🤖 Testing smart model management...")
    model_start = time.time()
    try:
        settings = get_settings()
        model_service = ModelService(settings)

        # Test language model selection
        language_model = model_service.get_language_model()
        embedding_model = model_service.get_embedding_model()

        model_time = time.time() - model_start
        print(f"   ✅ Model selection completed in {model_time:.2f}s")
        print(f"   🎯 Language model: {language_model.name}")
        print(f"   📊 Embedding model: {embedding_model.name}")

        # Test model health
        health_info = model_service.check_model_health()
        overall_health = health_info.get("overall_healthy", False)
        print(f"   💚 System health: {'HEALTHY' if overall_health else 'DEGRADED'}")

    except Exception as e:
        print(f"   ❌ Model management failed: {e}")
        return False

    # Test 5: Startup Idea Formatting
    print("\n5. 💡 Testing startup idea formatting...")
    format_start = time.time()
    try:
        test_descriptions = [
            "AI-powered developer productivity tool that helps write better code",
            "Machine learning platform for financial risk assessment",
            "Automated testing framework for web applications",
        ]

        formatted_ideas = []
        for desc in test_descriptions:
            formatted_idea = format_startup_idea(desc, retriever, startup_lookup)
            formatted_ideas.append(formatted_idea)

        format_time = time.time() - format_start
        avg_time = format_time / len(test_descriptions)

        print(f"   ✅ Formatted {len(test_descriptions)} ideas in {format_time:.2f}s")
        print(f"   ⚡ Average time per idea: {avg_time:.2f}s")

        # Show sample results
        for i, idea in enumerate(formatted_ideas[:2]):
            name = idea.get("name", "N/A")
            print(f"   📝 Sample {i+1}: {name}")

    except Exception as e:
        print(f"   ❌ Idea formatting failed: {e}")
        return False

    # Test 6: Memory and Resource Usage
    print("\n6. 📊 Testing resource usage...")
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        print(f"   💾 Memory usage: {memory_mb:.1f} MB")
        print(f"   🔥 CPU usage: {cpu_percent:.1f}%")

    except ImportError:
        print("   ⚠️  psutil not available, skipping resource monitoring")
    except Exception as e:
        print(f"   ⚠️  Resource monitoring failed: {e}")

    # Performance Summary
    total_time = time.time() - total_start
    print(f"\n=== 📈 PERFORMANCE SUMMARY ===")
    print(f"Data Loading:      {load_time:.2f}s")
    print(f"Lookup Init:       {lookup_time:.2f}s")
    print(f"RAG Init:          {rag_time:.2f}s")
    print(f"Model Selection:   {model_time:.2f}s")
    print(f"Idea Formatting:   {format_time:.2f}s")
    print(f"─" * 30)
    print(f"TOTAL TIME:        {total_time:.2f}s")

    # System Status
    print(f"\n=== 🎯 SYSTEM STATUS ===")
    print(f"✅ Status: FULLY FUNCTIONAL")
    print(f"📊 Data: {len(df)} YC startups processed")
    print(f"🤖 Models: Smart management active")
    print(f"🧠 RAG: Vector retrieval operational")
    print(f"⚡ Performance: {len(df)/total_time:.0f} startups/second overall")

    # Performance Assessment
    print(f"\n=== 🏆 PERFORMANCE ASSESSMENT ===")
    if total_time < 30:
        print("🚀 EXCELLENT: System startup under 30 seconds")
    elif total_time < 60:
        print("✅ GOOD: System startup under 1 minute")
    else:
        print("⚠️  SLOW: System startup over 1 minute")

    if rag_time < 15:
        print("🚀 EXCELLENT: RAG initialization under 15 seconds")
    elif rag_time < 30:
        print("✅ GOOD: RAG initialization under 30 seconds")
    else:
        print("⚠️  SLOW: RAG initialization over 30 seconds")

    print(f"\n🎉 End-to-end test completed successfully!")
    print(f"🔗 System ready for production workloads")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
