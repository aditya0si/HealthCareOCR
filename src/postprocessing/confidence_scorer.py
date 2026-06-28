import re
from src.utils.medical_dict import MedicalDictionary

def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

class ConfidenceScorer:
    def __init__(self, medical_dict: MedicalDictionary = None):
        self.dict = medical_dict if medical_dict is not None else MedicalDictionary()
        
        # Prepare data structures for fast fuzzy matching
        print("Preparing fast fuzzy matching search structures...")
        self.words_by_len = {}
        for w in self.dict.words:
            length = len(w)
            if length not in self.words_by_len:
                self.words_by_len[length] = []
            self.words_by_len[length].append((w, set(w)))
        print("Fuzzy structures ready.")

    def find_best_fuzzy_match(self, word: str) -> tuple[str, float]:
        """
        Finds the best fuzzy match in the dictionary for the given word.
        Returns (best_match_word, similarity_score).
        """
        w = word.lower()
        w_len = len(w)
        w_set = set(w)
        
        best_match = None
        min_dist = 999
        
        # Search lengths in priority order (same length first, then +/-1, then +/-2)
        search_lengths = []
        for diff in [0, -1, 1, -2, 2]:
            l = w_len + diff
            if l >= 1:
                search_lengths.append(l)
                
        for l in search_lengths:
            candidates = self.words_by_len.get(l, [])
            for cand_word, cand_set in candidates:
                # Fast pre-filtering: check character bag overlap
                # If the overlap of unique characters is too small, skip
                min_overlap = max(1, min(w_len, l) - 2)
                if len(w_set.intersection(cand_set)) < min_overlap:
                    continue
                    
                dist = edit_distance(w, cand_word)
                if dist < min_dist:
                    min_dist = dist
                    best_match = cand_word
                    if min_dist <= 1:
                        break
            if min_dist <= 1:
                break
                
        if best_match and min_dist <= 2:
            max_len = max(w_len, len(best_match))
            similarity = 1.0 - (min_dist / max_len)
            return best_match, similarity
            
        return "", 0.0

    def score_word(self, word: str) -> float:
        """
        Scores a single word confidence.
        1.0 for exact dictionary matches, numeric values, or dates.
        Fuzzy match similarity if match found, else 0.0.
        """
        # Exact match check
        if self.dict.contains(word):
            return 1.0
            
        # Clean word from punctuation for fuzzy check
        clean_word = word.strip().lower()
        clean_word = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', clean_word)
        if not clean_word:
            return 1.0
            
        # Skip small words/punctuation/numbers
        if len(clean_word) <= 2 or clean_word.isdigit():
            return 1.0
            
        # Skip non-alphabetic words (dosages, numbers, compound symbols, hyphens)
        if not clean_word.isalpha():
            return 1.0
            
        # Fuzzy match check
        _, similarity = self.find_best_fuzzy_match(clean_word)
        return similarity

    def process_text(self, text: str, threshold: float = 0.6) -> tuple[str, list[dict]]:
        """
        Splits text into words/tokens, scores each word, wraps low-confidence words with
        [UNCERTAIN] markers, and returns the processed text along with metadata.
        """
        if not text:
            return "", []
            
        # Use regex to find words (alphanumeric sequences) and non-word sequences
        tokens = re.split(r'(\s+|[^a-zA-Z0-9\-\']+)', text)
        
        output_tokens = []
        flagged_words = []
        
        for token in tokens:
            if not token:
                continue
                
            # If token is alphanumeric/word
            if re.match(r'^[a-zA-Z0-9\-\']+$', token):
                conf = self.score_word(token)
                if conf < threshold:
                    output_tokens.append(f"[UNCERTAIN]{token}")
                    flagged_words.append({
                        "original": token,
                        "confidence": conf
                    })
                else:
                    output_tokens.append(token)
            else:
                # Keep whitespace and punctuation as-is
                output_tokens.append(token)
                
        processed_text = "".join(output_tokens)
        return processed_text, flagged_words
