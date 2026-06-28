import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

class JsonStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self._count = 0
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._count += 1
        # Decimate checking to avoid heavy CPU tokenizer decode overhead
        # Only check every 5 tokens, and skip check if we generated less than 15 tokens
        if self._count < 15 or self._count % 5 != 0:
            return False
            
        generated_ids = input_ids[0][self.prompt_length:]
        if len(generated_ids) == 0:
            return False
            
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        stripped = decoded.strip()
        
        # Condition 1: Balanced braces
        if '{' in stripped:
            open_braces = stripped.count('{')
            close_braces = stripped.count('}')
            if open_braces > 0 and open_braces == close_braces:
                return True
                
        # Condition 2: Newline after first '{'
        first_brace = decoded.find('{')
        if first_brace != -1:
            if '\n' in decoded[first_brace:]:
                return True
                
        return False

class ClinicalSummarizer:
    BASE_MODEL_ID = "microsoft/Phi-4-mini-instruct"

    def __init__(self, model=None, tokenizer=None, model_id=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id or self.BASE_MODEL_ID
        self.device = device

    def load_model(self):
        if self.model is not None:
            return
            
        print(f"Loading LLM {self.model_id} for Summarization in 4-bit...")
        from transformers import BitsAndBytesConfig
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            attn_implementation="sdpa"
        )
        self.model.eval()

    def summarize_text(self, text: str) -> dict:
        if not text:
            return {}
            
        self.load_model()
        
        # Build prompt for structured medical summary extraction
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical report analyzer. "
                    "Extract structured clinical information from the provided OCR medical report. "
                    "Return ONLY a flat, compressed, single-line JSON object matching the schema below. "
                    "OMIT any keys if their value is null, empty string, or an empty list. "
                    "Do NOT pretty-print, add indentation, or include newlines in the JSON. Output a single flat line. "
                    "Keep all output values as concise as possible. "
                    "Keep all text values (names, findings, diagnoses, test names) to a maximum of 5 words per value. Do not generate long lists of tests or medical procedures. "
                    "Do not include any explanations, introduction, markdown codeblocks, or conversational filler. "
                    "Ensure JSON fields are exactly named as specified if included.\n\n"
                    "JSON Schema:\n"
                    "{\n"
                    "  \"patient_name\": \"string\",\n"
                    "  \"age_sex\": \"string\",\n"
                    "  \"document_type\": \"lab_report | prescription | discharge_summary | other\",\n"
                    "  \"date\": \"string\",\n"
                    "  \"hospital\": \"string\",\n"
                    "  \"doctor\": \"string\",\n"
                    "  \"key_findings\": [\"string\"],\n"
                    "  \"medications\": [{\"drug\": \"string\", \"dosage\": \"string\", \"frequency\": \"string\"}],\n"
                    "  \"diagnoses\": [\"string\"],\n"
                    "  \"abnormal_values\": [{\"test\": \"string\", \"value\": \"string\", \"reference\": \"string\", \"status\": \"high | low | abnormal\"}],\n"
                    "  \"summary\": \"one sentence clinical overview (max 12 words)\"\n"
                    "}"
                )
            },
            {
                "role": "user",
                "content": f"OCR Text:\n\n{text}"
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList([JsonStoppingCriteria(self.tokenizer, input_len)])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.0,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.15,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|end|>")],
                stopping_criteria=stopping_criteria
            )
            
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Parse JSON
        parsed = self._parse_json_response(response)
        
        # Restore schema structure programmatically
        default_schema = {
            "patient_name": None,
            "age_sex": None,
            "document_type": "other",
            "date": None,
            "hospital": None,
            "doctor": None,
            "key_findings": [],
            "medications": [],
            "diagnoses": [],
            "abnormal_values": [],
            "summary": None
        }
        
        restored = {}
        for key, default in default_schema.items():
            val = parsed.get(key, None)
            if val is None or val == "" or val == []:
                restored[key] = default
            elif isinstance(default, list) and not isinstance(val, list):
                restored[key] = [val]
            else:
                restored[key] = val
                
        # Also copy other keys (e.g. error, raw_response) if present
        for key, val in parsed.items():
            if key not in restored:
                restored[key] = val
                
        return restored

    def _parse_json_response(self, response: str) -> dict:
        # Try to find JSON block using regex
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, response, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            # Fallback: find first '{' and last '}'
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                json_str = response[start:end+1]
            else:
                json_str = response
                
        # Repair common JSON syntax errors from LLMs
        json_str = json_str.strip()
        # 1. Replace key = value (or key="value", key=123) with key : value
        json_str = re.sub(r'("[^"]*")\s*=\s*(["{\[\d\w])', r'\1:\2', json_str)
        json_str = re.sub(r'("[^"]*")\s*=\s*(true|false|null)', r'\1:\2', json_str)
        
        # 2. Fix trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*\}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        # 3. Balance curly braces to drop any extra trailing curly braces (e.g. }} instead of })
        start_brace = json_str.find("{")
        if start_brace != -1:
            json_str = json_str[start_brace:]
            open_braces = 0
            in_string = False
            escape = False
            cut_idx = -1
            
            for idx, char in enumerate(json_str):
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                        if open_braces == 0:
                            cut_idx = idx + 1
                            break
            if cut_idx != -1:
                json_str = json_str[:cut_idx]
                
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing generated JSON summary: {e}")
            print(f"Cleaned JSON string: {json_str}")
            print(f"Raw model response: {response}")
            return {
                "error": "Failed to parse clinical summary JSON",
                "raw_response": response
            }
