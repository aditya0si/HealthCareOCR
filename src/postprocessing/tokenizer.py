from transformers import AutoTokenizer

class MedicalTokenizer:
    def __init__(self, model_id: str = "microsoft/Phi-4-mini-instruct"):
        """
        Wrapper around the pretrained LLM tokenizer, optimized for medical subword segmentation.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(self, text: str) -> list[str]:
        """
        Segments text into subword tokens.
        """
        return self.tokenizer.tokenize(text)

    def encode(self, text: str) -> list[int]:
        """
        Encodes text into token IDs.
        """
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decodes token IDs back to a string.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)
