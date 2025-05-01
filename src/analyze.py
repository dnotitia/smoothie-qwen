import os
import json
import pickle
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any


class TokenAnalyzer:
    """
    Analyzes token combinations to identify patterns that may produce target language.
    """
    
    def __init__(self, 
                 model_name: str,
                 tokenizer: Any,
                 target_tokens: List[int],
                 broken_tokens: List[int],
                 unicode_ranges: List[Tuple[int, int]],
                 cache_dir: str = ".token_cache",
                 verbose: bool = False):
        """
        Initialize the token analyzer.
        
        Args:
            model_name: Model name or path
            tokenizer: Tokenizer instance
            target_tokens: List of token IDs containing target characters
            broken_tokens: List of token IDs containing broken characters
            unicode_ranges: List of Unicode ranges to target [(start, end), ...]
            cache_dir: Directory to cache token analysis
            verbose: Enable verbose logging
        """
        # Model information
        self.model_name = model_name
        self.tokenizer = tokenizer
        
        # Token lists
        self.target_tokens = target_tokens
        self.broken_tokens = broken_tokens
        
        # Unicode ranges
        self.unicode_ranges = unicode_ranges
        
        # Analysis results
        self.token_analysis = {}  # {token_id: {n: n-gram probability, ...}}
        
        # Cache setup
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache file paths
        model_name_safe = self.model_name.replace('/', '_')
        self.token_analysis_cache = os.path.join(self.cache_dir, f"{model_name_safe}_token_analysis.pkl")
        
        # Set up logging
        self.logger = logging.getLogger("TokenAnalyzer")
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    def is_target_char(self, char: str) -> bool:
        """Check if a character is in the target Unicode ranges."""
        if not char or char.isspace() or char.isdigit() or char.isascii():
            return False
        
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.unicode_ranges)
    
    def has_target_char(self, text: str) -> bool:
        """Check if text contains any characters in target Unicode ranges."""
        return any(self.is_target_char(char) for char in text)
    
    def analyze_ngram_combinations(self, 
                                  sample_size: int = 100, 
                                  max_ngram: int = 4) -> Dict[int, Dict[int, float]]:
        """
        Analyze n-gram combinations of broken tokens to estimate their likelihood
        of producing target language.
        
        Args:
            sample_size: Number of token combinations to sample per token
            max_ngram: Maximum n-gram size to analyze (2-4)
            
        Returns:
            Dictionary of token analysis {token_id: {n: probability, ...}}
        """
        self.logger.info(f"Starting n-gram analysis (sample size: {sample_size}, max n-gram: {max_ngram})...")
        
        # Initialize analysis results
        self.token_analysis = {}
        
        # Analyze each broken token
        for broken_token in tqdm(self.broken_tokens, desc="Analyzing broken tokens"):
            self.token_analysis[broken_token] = {}
            
            # Sample candidate tokens for analysis
            if len(self.broken_tokens) <= sample_size:
                candidate_tokens = np.array(self.broken_tokens[:sample_size])
                sample_size = len(candidate_tokens)
            else:
                candidate_tokens = np.random.choice(self.broken_tokens, sample_size, replace=False)
            
            # Perform n-gram analysis
            if max_ngram >= 2:
                self._analyze_ngram(broken_token, candidate_tokens, n=2, sample_size=sample_size)
            
            if max_ngram >= 3:
                self._analyze_ngram(broken_token, candidate_tokens, n=3, sample_size=sample_size)
                
            if max_ngram >= 4:
                self._analyze_ngram(broken_token, candidate_tokens, n=4, sample_size=sample_size)
        
        self.logger.info(f"N-gram analysis completed for {len(self.token_analysis)} tokens")
        return self.token_analysis
    
    def _analyze_ngram(self, token: int, candidates: np.ndarray, n: int, sample_size: int) -> None:
        """
        Analyze n-gram combinations for a specific token.
        
        Args:
            token: Token ID to analyze
            candidates: Array of candidate token IDs to combine with
            n: N-gram size
            sample_size: Number of samples to analyze
        """
        target_count = 0
        total_count = 0

        # Adjust sample size based on n-gram size
        if n == 2:
            # 2-gram uses full sample size
            adjusted_sample = sample_size
            combinations = [(token, int(next_token)) for next_token in 
                           np.random.choice(candidates, adjusted_sample, replace=False)]
        elif n == 3:
            # 3-gram uses square root of sample size
            adjusted_sample = int(np.sqrt(sample_size))
            combinations = []
            for next_token1 in np.random.choice(candidates, adjusted_sample, replace=False):
                for next_token2 in np.random.choice(candidates, adjusted_sample, replace=False):
                    combinations.append((token, int(next_token1), int(next_token2)))
        elif n == 4:
            # 4-gram uses cube root of sample size
            adjusted_sample = max(2, int(np.cbrt(sample_size)))
            combinations = []
            for next_token1 in np.random.choice(candidates, adjusted_sample, replace=False):
                for next_token2 in np.random.choice(candidates, adjusted_sample, replace=False):
                    for next_token3 in np.random.choice(candidates, adjusted_sample, replace=False):
                        combinations.append((token, int(next_token1), int(next_token2), int(next_token3)))
        
        # Analyze each combination
        for combo in combinations:
            ngram_text = self.tokenizer.decode(list(combo))
            if self.has_target_char(ngram_text):
                target_count += 1
            total_count += 1
        
        # Calculate and store probability
        probability = target_count / total_count if total_count > 0 else 0
        self.token_analysis[token][n] = probability
        self.logger.debug(f"Token {token} {n}-gram analysis: {target_count}/{total_count} = {probability:.4f}")
    
    def save_token_data(self) -> None:
        """Save token analysis to cache file."""
        self.logger.info(f"Saving token analysis data...")
        
        # Save token analysis (using pickle for complex data structure)
        with open(self.token_analysis_cache, 'wb') as f:
            pickle.dump(self.token_analysis, f)
        
        self.logger.info(f"Token analysis data saved to: {self.token_analysis_cache}")
    
    def load_token_data(self) -> bool:
        """
        Load token analysis from cache file.
        
        Returns:
            bool: True if cache was successfully loaded, False otherwise
        """
        # Check if cache file exists
        if not os.path.exists(self.token_analysis_cache):
            self.logger.info("No cached token analysis found")
            return False
        
        try:
            # Load token analysis
            with open(self.token_analysis_cache, 'rb') as f:
                self.token_analysis = pickle.load(f)
            
            # Validate loaded data
            if not isinstance(self.token_analysis, dict):
                raise ValueError("Invalid token analysis format in cache")
            
            self.logger.info(f"Loaded token analysis from cache")
            self.logger.info(f"- Analyzed tokens: {len(self.token_analysis):,}")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load token analysis from cache: {e}")
            return False