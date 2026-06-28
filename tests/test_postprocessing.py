import os
import pytest
import torch
import time
from src.utils.medical_dict import MedicalDictionary
from src.postprocessing.confidence_scorer import ConfidenceScorer
from src.postprocessing.llm_corrector import LLMCorrector
from src.utils.gpu_manager import GPUManager

@pytest.fixture(scope="module")
def dictionary():
    # Will download and cache if not present, then load
    return MedicalDictionary()

@pytest.fixture(scope="module")
def scorer(dictionary):
    return ConfidenceScorer(dictionary)

def test_medical_dictionary_loading(dictionary):
    # Verify the dictionary is populated
    assert len(dictionary.words) > 50000
    
    # Test standard medical words
    assert dictionary.contains("ibuprofen")
    assert dictionary.contains("metformin")
    assert dictionary.contains("paracetamol")
    
    # Test custom abbreviations & units
    assert dictionary.contains("cbc")
    assert dictionary.contains("hb")
    assert dictionary.contains("mg")
    assert dictionary.contains("ml")
    assert dictionary.contains("bp")
    assert dictionary.contains("spo2")
    
    # Test normalization of punctuation and case
    assert dictionary.contains("Ibuprofen,")
    assert dictionary.contains("(metformin)")
    assert dictionary.contains("CBC!")
    
    # Test auto-accept values (numbers and dates)
    assert dictionary.contains("400")
    assert dictionary.contains("12.5")
    assert dictionary.contains("2026-06-26")
    assert dictionary.contains("12/16/2024")

def test_confidence_scoring(scorer):
    # High confidence for exact match
    assert scorer.score_word("Ibuprofen") == 1.0
    assert scorer.score_word("Patient") == 1.0
    assert scorer.score_word("400") == 1.0
    assert scorer.score_word("BID") == 1.0
    
    # Fuzzy match confidence (Levenshtein distance <= 2)
    conf_ibu = scorer.score_word("ibuprufen") # len 9, dist 1
    assert conf_ibu > 0.8
    assert conf_ibu < 1.0
    
    # No match confidence for random strings
    assert scorer.score_word("xyzqprw") < 0.6
    assert scorer.score_word("qwertyuiop") < 0.6

def test_text_processing_markers(scorer):
    # Use a truly garbled word that won't fuzzy-match anything in the dictionary
    # "ibuprufen" scores 0.89 (1 edit from "ibuprofen") — correctly NOT flagged
    raw_text = "The patient was prescribed xbqzrufen 400 mg and metformin."
    processed, flagged = scorer.process_text(raw_text)
    
    # Verify that the garbled word is flagged with [UNCERTAIN]
    assert "[UNCERTAIN]xbqzrufen" in processed
    # "patient", "prescribed", "400", "mg", "metformin" should NOT be flagged
    assert "[UNCERTAIN]patient" not in processed
    assert "[UNCERTAIN]metformin" not in processed
    assert "[UNCERTAIN]prescribed" not in processed
    
    assert len(flagged) >= 1
    garbled_entry = [f for f in flagged if f["original"] == "xbqzrufen"]
    assert len(garbled_entry) == 1
    assert garbled_entry[0]["confidence"] < 0.6
    
    # Also verify that near-matches are correctly NOT flagged (high similarity)
    near_match_text = "The patient was prescribed ibuprufen 400 mg."
    processed2, flagged2 = scorer.process_text(near_match_text)
    # ibuprufen is 1 edit distance from ibuprofen → score ~0.89 → above threshold → not flagged
    assert "[UNCERTAIN]ibuprufen" not in processed2


@pytest.mark.skip(reason="LLM correction is disintegrated/disabled to avoid CUDA memory issues")
def test_llm_corrector_lifecycle_and_execution():
    corrector = LLMCorrector()
    
    # Measure VRAM before load
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Run correction (cold start — includes model loading)
    input_text = "The patient is a 51 years old male. Prescribed: [UNCERTAIN]ibuprufen 400 [UNCERTAIN]mg BID."
    
    start_time = time.time()
    corrected_text = corrector.correct_text(input_text)
    elapsed = time.time() - start_time
    
    # Measure VRAM after load
    vram_loaded = torch.cuda.memory_allocated() / (1024 ** 2)
    
    print(f"\n[LLM Corrector Test]")
    print(f"  Input: {input_text}")
    print(f"  Output: {corrected_text}")
    print(f"  Cold Start Latency: {elapsed:.2f} seconds")
    print(f"  VRAM Loaded: {vram_loaded:.2f} MB")
    
    # Assertions on correction quality
    assert "ibuprofen" in corrected_text.lower()
    assert "[uncertain]" not in corrected_text.lower()
    
    # Hot latency test — model is STILL loaded from the cold call above
    start_hot = time.time()
    hot_corrected = corrector.correct_text(input_text)
    hot_elapsed = time.time() - start_hot
    print(f"  Hot Latency: {hot_elapsed:.2f} seconds")
    assert hot_elapsed < 5.0, f"Hot correction latency too high: {hot_elapsed:.2f}s"
    assert "ibuprofen" in hot_corrected.lower()
    
    # Now unload model and verify VRAM cleanup
    corrector.unload()
    torch.cuda.empty_cache()
    
    # Measure VRAM after unload
    vram_after = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"  VRAM After Unload: {vram_after:.2f} MB")
    
    # VRAM should be released back to near-baseline level (allow some CUDA context overhead)
    # The active weights of the model (~2GB) must be completely unloaded
    assert (vram_loaded - vram_after) > 1500.0, "Model weights were not unloaded from VRAM"
