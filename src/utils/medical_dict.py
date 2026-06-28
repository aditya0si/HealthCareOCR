import os
import urllib.request
import re

class MedicalDictionary:
    ENGLISH_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    MEDICAL_URL = "https://raw.githubusercontent.com/Glutanimate/wordlist-medicalterms-en/master/wordlist.txt"
    
    def __init__(self, resources_dir: str = "resources"):
        self.resources_dir = resources_dir
        self.english_path = os.path.join(resources_dir, "words_english.txt")
        self.medical_path = os.path.join(resources_dir, "words_medical.txt")
        self.words = set()
        
        # Ensure resources directory exists
        os.makedirs(self.resources_dir, exist_ok=True)
        
        # Load or download dictionary files
        self._load_dictionary()
        self._add_abbreviations_and_units()

    def _download_file(self, url: str, dest_path: str):
        print(f"Downloading {url} to {dest_path}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    def _load_dictionary(self):
        # 1. Download if missing
        if not os.path.exists(self.english_path):
            self._download_file(self.ENGLISH_URL, self.english_path)
        if not os.path.exists(self.medical_path):
            self._download_file(self.MEDICAL_URL, self.medical_path)

        # 2. Parse and load English words
        if os.path.exists(self.english_path):
            print(f"Loading English words from {self.english_path}...")
            with open(self.english_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        self.words.add(w)
                        
        # 3. Parse and load Medical words
        if os.path.exists(self.medical_path):
            print(f"Loading Medical words from {self.medical_path}...")
            with open(self.medical_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        # Split multi-word terms into individual words for word-level matching
                        parts = re.findall(r'[a-zA-Z0-9]+', w)
                        for part in parts:
                            self.words.add(part)

        print(f"Lexicon loaded: {len(self.words)} unique words.")

    def _add_abbreviations_and_units(self):
        # Common Indian medical abbreviations, symbols, units, and hospital report terms
        custom_terms = {
            # Clinical abbreviations
            "bp", "pr", "rr", "hr", "spo2", "temp", "gcs", "cvs", "pa", "cns", "rs",
            "bid", "tid", "qid", "od", "hs", "prn", "qds", "bd", "tds", "po", "ac", "pc",
            "tab", "cap", "inj", "syp", "susp", "soln", "cre", "oint", "dr", "sis",
            
            # Labs & parameters
            "cbc", "lft", "kft", "rft", "hb", "tlc", "dlc", "plt", "rbc", "wbc", "esr",
            "fbs", "ppbs", "rbs", "sgot", "sgpt", "ast", "alt", "alp", "tsh", "t3", "t4",
            "bun", "cr", "ua", "usg", "ecg", "eeg", "ct", "mri", "xr", "xray",
            "hba1c", "hgb", "ldh", "crp", "ra", "aso", "widal", "typhi", "dengue", "ns1",
            
            # Units
            "mg", "mcg", "g", "kg", "ml", "l", "dl", "fl", "pg", "ug", "iu", "u", "meq",
            "mmol", "umol", "cells", "cumm", "hpf", "lpf", "gm", "percent", "pct",
            "min", "hr", "sec", "beats", "bpm", "mmhg", "c", "f",
            
            # Common Indian report terminology
            "patient", "age", "gender", "sex", "date", "ref", "referred", "dr", "doctor",
            "hospital", "clinic", "lab", "laboratory", "report", "summary", "discharge",
            "admission", "history", "examination", "diagnosis", "treatment", "complaints",
            "findings", "investigations", "medication", "medications", "rx", "dx", "hx",
            "normal", "abnormal", "range", "units", "result", "results", "value", "values",
            "investigation", "prescribed", "tablet", "tablets", "capsule", "capsules",
            
            # Common Indian names/regions in dataset
            "kastoor", "patient_kastoor", "meadowview", "gomez", "brown", "gita", "devi",
            "sharma", "singh", "kumar", "patel", "verma", "gupta", "khan", "reddy", "rao",
            "choudhary", "prasad", "yadav", "joshi", "nair", "pillai", "menon"
        }
        for term in custom_terms:
            self.words.add(term.lower())
            
    def contains(self, word: str) -> bool:
        """
        Returns True if the word exists in the dictionary.
        Word is normalized (cleaned of punctuation, lowercased).
        """
        # Strip trailing punctuation
        w = word.strip().lower()
        w = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', w)
        if not w:
            return True # Empty or only punctuation is accepted
            
        # Ignore numeric/date values
        if re.match(r'^\d+(\.\d+)?$', w) or re.match(r'^\d{1,4}[-/\.]\d{1,4}[-/\.]\d{1,4}$', w):
            return True
            
        return w in self.words
