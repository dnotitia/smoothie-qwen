import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Tuple, Dict, Any


class Identifier:
    """
    Identifies tokens in the target Unicode ranges, including broken tokens.
    """
    
    # Default broken character Unicode
    BROKEN_CHAR = '�'  # U+FFFD

    def __init__(self, 
                 model_name: str,
                 unicode_ranges: List[Tuple[int, int]] = None,
                 cache_dir: str = ".token_cache",
                 model_dtype: str = "bfloat16",
                 verbose: bool = False):
        """
        Initialize the token identifier.
        
        Args:
            model_name: Model name or path
            unicode_ranges: List of Unicode ranges to target [(start, end), ...]
            cache_dir: Directory to cache token analysis
            verbose: Enable verbose logging
        """
        # Model information
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.model_dtype = model_dtype
        
        # Unicode ranges to target
        self.unicode_ranges = unicode_ranges
        
        # Identified tokens
        self.target_tokens = []  # Tokens containing characters in target Unicode ranges
        self.broken_tokens = []  # Tokens containing broken characters (�)
        
        # Cache setup
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache file paths
        model_name_safe = self.model_name.replace('/', '_')
        self.target_tokens_cache = os.path.join(self.cache_dir, f"{model_name_safe}_target_tokens.json")
        self.broken_tokens_cache = os.path.join(self.cache_dir, f"{model_name_safe}_broken_tokens.json")
        
        # Set up logging
        self.logger = logging.getLogger("Identifier")
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=self.model_dtype,
        )
        
        # Check model data type
        lm_head_dtype = self.model.lm_head.weight.dtype
        self.logger.info(f"Model loaded: vocab size {self.tokenizer.vocab_size:,} tokens, dtype: {lm_head_dtype}")
    
    def is_target_char(self, char: str) -> bool:
        """Check if a character is in the target Unicode ranges."""
        if not char or char.isspace() or char.isdigit() or char.isascii():
            return False
        
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.unicode_ranges)
    
    def has_target_char(self, text: str) -> bool:
        """Check if text contains any characters in target Unicode ranges."""
        return any(self.is_target_char(char) for char in text)
    
    def has_broken_char(self, text: str) -> bool:
        """Check if text contains any broken characters (�)."""
        return self.BROKEN_CHAR in text
    
    def identify_tokens(self) -> Tuple[List[int], List[int]]:
        """
        Identify tokens containing target characters or broken characters.
        
        Returns:
            Tuple of (target_tokens, broken_tokens)
        """
        self.logger.info("Starting token identification...")
        vocab_size = self.tokenizer.vocab_size
        
        # Reset result lists
        self.target_tokens = []
        self.broken_tokens = []
        
        # Analyze all tokens in vocabulary
        for token_id in tqdm(range(vocab_size), desc="Analyzing tokens"):
            token_text = self.tokenizer.decode([token_id])
            
            # Check for target characters
            if self.has_target_char(token_text):
                self.target_tokens.append(token_id)
            
            # Check for broken characters
            if self.has_broken_char(token_text):
                self.broken_tokens.append(token_id)
        
        self.logger.info(f"Identified target tokens: {len(self.target_tokens):,} ({len(self.target_tokens)/vocab_size:.2%})")
        self.logger.info(f"Identified broken tokens: {len(self.broken_tokens):,} ({len(self.broken_tokens)/vocab_size:.2%})")
        
        return self.target_tokens, self.broken_tokens
    
    def save_token_data(self) -> None:
        """Save identified tokens to cache files."""
        self.logger.info(f"Saving token data to cache...")
        
        # Save target tokens
        with open(self.target_tokens_cache, 'w', encoding='utf-8') as f:
            json.dump(self.target_tokens, f)
        
        # Save broken tokens
        with open(self.broken_tokens_cache, 'w', encoding='utf-8') as f:
            json.dump(self.broken_tokens, f)
        
        self.logger.info(f"Token data saved to cache: {self.cache_dir}")
    
    def load_token_data(self) -> bool:
        """
        Load identified tokens from cache files.
        
        Returns:
            bool: True if cache was successfully loaded, False otherwise
        """
        # Check if cache files exist
        if not (os.path.exists(self.target_tokens_cache) and 
                os.path.exists(self.broken_tokens_cache)):
            self.logger.info("No cached token data found")
            return False
        
        try:
            # Load target tokens
            with open(self.target_tokens_cache, 'r', encoding='utf-8') as f:
                self.target_tokens = json.load(f)
            
            # Load broken tokens
            with open(self.broken_tokens_cache, 'r', encoding='utf-8') as f:
                self.broken_tokens = json.load(f)
            
            # Validate loaded data
            if not isinstance(self.target_tokens, list) or not isinstance(self.broken_tokens, list):
                raise ValueError("Invalid token data format in cache")
            
            self.logger.info(f"Loaded token data from cache")
            self.logger.info(f"- Target tokens: {len(self.target_tokens):,}")
            self.logger.info(f"- Broken tokens: {len(self.broken_tokens):,}")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load token data from cache: {e}")
            return False