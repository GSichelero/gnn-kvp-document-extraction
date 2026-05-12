#!/usr/bin/env python3
"""
Semantic Normalization Training and Testing
Focuses on normalizing extracted values to standard formats
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
from datetime import datetime
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticNormalizationAnalyzer:
    """Analyzes semantic normalization patterns"""
    
    def __init__(self, annotation_dir: str):
        self.annotation_dir = Path(annotation_dir)
        self.data = []
        self.patterns = defaultdict(list)
        self.normalizations = defaultdict(list)
    
    def load_annotations(self):
        """Load annotations"""
        for json_file in self.annotation_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data.append(data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.data)} annotation files")
        return self.data
    
    def analyze_normalization_needs(self) -> Dict:
        """Analyze what needs normalization"""
        analysis = {
            'date_values': defaultdict(int),
            'amount_values': defaultdict(int),
            'numeric_values': defaultdict(int),
            'text_values': defaultdict(int),
            'normalization_cases': defaultdict(list),
        }
        
        for doc in self.data:
            kvp_pairs = doc.get('kvp_pairs', [])
            
            for pair in kvp_pairs:
                key_text = pair.get('key', {}).get('text', '').lower()
                value_text = pair.get('value', {}).get('text', '').strip()
                
                # Classify and collect normalization needs
                if any(x in key_text for x in ['data', 'vencimento', 'competência']):
                    analysis['date_values'][value_text] += 1
                    analysis['normalization_cases']['dates'].append(value_text)
                
                elif any(x in key_text for x in ['valor', 'total', 'subtotal']):
                    analysis['amount_values'][value_text] += 1
                    analysis['normalization_cases']['amounts'].append(value_text)
                
                elif re.match(r'^\d+$', value_text):
                    analysis['numeric_values'][value_text] += 1
                    analysis['normalization_cases']['numbers'].append(value_text)
                
                else:
                    analysis['text_values'][value_text] += 1
                    analysis['normalization_cases']['text'].append(value_text)
        
        return analysis
    
    def suggest_normalization_rules(self) -> Dict:
        """Suggest normalization rules"""
        analysis = self.analyze_normalization_needs()
        
        rules = {
            'dates': self._extract_date_patterns(
                analysis['normalization_cases'].get('dates', [])
            ),
            'amounts': self._extract_amount_patterns(
                analysis['normalization_cases'].get('amounts', [])
            ),
            'numbers': self._extract_number_patterns(
                analysis['normalization_cases'].get('numbers', [])
            ),
        }
        
        return rules
    
    def _extract_date_patterns(self, dates: List[str]) -> Dict:
        """Extract date patterns"""
        patterns = {}
        
        date_regex_patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{2,4})', 'dd/mm/yyyy'),
            (r'(\d{1,2})-(\d{1,2})-(\d{2,4})', 'dd-mm-yyyy'),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'yyyy-mm-dd'),
            (r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', 'name dd yyyy'),
        ]
        
        for date_str in dates:
            for pattern, name in date_regex_patterns:
                if re.search(pattern, date_str):
                    patterns[name] = patterns.get(name, 0) + 1
        
        return patterns
    
    def _extract_amount_patterns(self, amounts: List[str]) -> Dict:
        """Extract amount patterns"""
        patterns = {}
        
        amount_regex_patterns = [
            (r'R\$\s*([\d.,]+)', 'R$ format'),
            (r'([\d.,]+)$', 'numeric format'),
            (r'([\d.]+,\d{2})', 'point separator'),
            (r'([\d,]+\.\d{2})', 'comma separator'),
        ]
        
        for amount_str in amounts:
            for pattern, name in amount_regex_patterns:
                if re.search(pattern, amount_str):
                    patterns[name] = patterns.get(name, 0) + 1
        
        return patterns
    
    def _extract_number_patterns(self, numbers: List[str]) -> Dict:
        """Extract number patterns"""
        stats = {
            'total': len(numbers),
            'unique': len(set(numbers)),
            'avg_length': np.mean([len(n) for n in numbers]),
            'max_length': max([len(n) for n in numbers]) if numbers else 0,
            'min_length': min([len(n) for n in numbers]) if numbers else 0,
        }
        
        return stats


class SemanticNormalizer:
    """Performs semantic normalization on extracted values"""
    
    def __init__(self):
        self.patterns = {}
        self.normalization_cache = {}
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to ISO format (YYYY-MM-DD)"""
        if not date_str or not isinstance(date_str, str):
            return None
        
        date_str = date_str.strip()
        
        # Cache
        if date_str in self.normalization_cache:
            return self.normalization_cache[date_str]
        
        # Try common Brazilian date formats
        patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m[3]}-{m[2]:0>2}-{m[1]:0>2}"),
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: f"{m[3]}-{m[2]:0>2}-{m[1]:0>2}"),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m[1]}-{m[2]:0>2}-{m[3]:0>2}"),
        ]
        
        for pattern, formatter in patterns:
            match = re.match(pattern, date_str)
            if match:
                groups = match.groups()
                result = formatter(groups)
                self.normalization_cache[date_str] = result
                return result
        
        return None
    
    def normalize_amount(self, amount_str: str) -> Optional[float]:
        """Normalize amount to float"""
        if not amount_str or not isinstance(amount_str, str):
            return None
        
        if amount_str in self.normalization_cache:
            return self.normalization_cache[amount_str]
        
        amount_str = amount_str.strip()
        
        # Remove R$ and spaces
        amount_str = re.sub(r'R\$\s*', '', amount_str).strip()
        
        # Handle different decimal separators used in Brazil
        # If both . and , exist, the last one is decimal
        if '.' in amount_str and ',' in amount_str:
            last_dot = amount_str.rfind('.')
            last_comma = amount_str.rfind(',')
            
            if last_comma > last_dot:
                # comma is decimal separator
                amount_str = amount_str.replace('.', '').replace(',', '.')
            else:
                # dot is decimal separator
                amount_str = amount_str.replace(',', '')
        elif ',' in amount_str:
            # Only comma - likely European/Brazilian style
            if amount_str.count(',') == 1:
                amount_str = amount_str.replace(',', '.')
            else:
                # Multiple commas - comma is thousands
                amount_str = amount_str.replace(',', '')
        
        try:
            result = float(amount_str)
            self.normalization_cache[amount_str] = result
            return result
        except ValueError:
            return None
    
    def normalize_numeric(self, value_str: str) -> Optional[int]:
        """Normalize to integer"""
        if not value_str or not isinstance(value_str, str):
            return None
        
        if value_str in self.normalization_cache:
            return self.normalization_cache[value_str]
        
        try:
            result = int(re.sub(r'\D', '', value_str))
            self.normalization_cache[value_str] = result
            return result
        except ValueError:
            return None
    
    def normalize_text(self, text_str: str) -> str:
        """Normalize text"""
        if not isinstance(text_str, str):
            return str(text_str)
        
        # Normalize whitespace
        text_str = re.sub(r'\s+', ' ', text_str.strip())
        
        # Remove extra punctuation
        text_str = re.sub(r'[^\w\s\-\.,/]', '', text_str)
        
        return text_str
    
    def normalize_value(self, value_str: str, value_type: str) -> Dict:
        """Normalize value based on type"""
        result = {
            'original': value_str,
            'normalized': None,
            'type': value_type,
            'confidence': 0.0,
        }
        
        try:
            if value_type == 'date':
                normalized = self.normalize_date(value_str)
                if normalized:
                    result['normalized'] = normalized
                    result['confidence'] = 0.95
            
            elif value_type == 'amount':
                normalized = self.normalize_amount(value_str)
                if normalized is not None:
                    result['normalized'] = normalized
                    result['confidence'] = 0.95
            
            elif value_type == 'numeric':
                normalized = self.normalize_numeric(value_str)
                if normalized is not None:
                    result['normalized'] = normalized
                    result['confidence'] = 0.95
            
            else:
                normalized = self.normalize_text(value_str)
                if normalized:
                    result['normalized'] = normalized
                    result['confidence'] = 0.8
        
        except Exception as e:
            logger.debug(f"Normalization error for '{value_str}': {e}")
        
        return result


class SemanticNormalizationModel(nn.Module):
    """Neural model for semantic normalization classification"""
    
    def __init__(self, 
                 vocab_size: int = 5000,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_outputs: int = 4):  # date, amount, numeric, text
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        # Embedding
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, 
                                     key_padding_mask=None)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class ComprehensiveSemanticTestSuite:
    """Comprehensive semantic normalization tests"""
    
    def __init__(self, output_dir: str = 'semantic_normalization_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_all_tests(self, annotation_dir: str):
        """Run all semantic normalization tests"""
        logger.info("\n" + "="*80)
        logger.info("SEMANTIC NORMALIZATION TEST SUITE")
        logger.info("="*80)
        
        # Test 1: Analysis
        logger.info("\n" + "-"*80)
        logger.info("TEST 1: Normalization Needs Analysis")
        logger.info("-"*80)
        
        analyzer = SemanticNormalizationAnalyzer(annotation_dir)
        analyzer.load_annotations()
        
        analysis = analyzer.analyze_normalization_needs()
        rules = analyzer.suggest_normalization_rules()
        
        logger.info(f"Found {len(analyzer.data)} documents to analyze")
        logger.info(f"\nDate values: {sum(analysis['date_values'].values())}")
        logger.info(f"Amount values: {sum(analysis['amount_values'].values())}")
        logger.info(f"Numeric values: {sum(analysis['numeric_values'].values())}")
        logger.info(f"Text values: {sum(analysis['text_values'].values())}")
        
        logger.info("\nDate patterns detected:")
        for pattern, count in rules['dates'].items():
            logger.info(f"  {pattern}: {count}")
        
        logger.info("\nAmount patterns detected:")
        for pattern, count in rules['amounts'].items():
            logger.info(f"  {pattern}: {count}")
        
        self.results['analysis'] = {
            'value_counts': {
                'dates': sum(analysis['date_values'].values()),
                'amounts': sum(analysis['amount_values'].values()),
                'numbers': sum(analysis['numeric_values'].values()),
                'text': sum(analysis['text_values'].values()),
            },
            'patterns': rules,
        }
        
        # Test 2: Normalization accuracy
        logger.info("\n" + "-"*80)
        logger.info("TEST 2: Normalization Accuracy")
        logger.info("-"*80)
        
        normalizer = SemanticNormalizer()
        self._test_normalization_accuracy(
            analyzer, normalizer, analysis
        )
        
        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {self.output_dir}")
    
    def _test_normalization_accuracy(self, 
                                     analyzer,
                                     normalizer,
                                     analysis):
        """Test normalization accuracy"""
        
        test_cases = {
            'dates': [
                ('01/02/2023', '2023-02-01'),
                ('15-05-2022', '2022-05-15'),
                ('2023-12-31', '2023-12-31'),
            ],
            'amounts': [
                ('R$ 1.000,50', 1000.50),
                ('1.200.000,00', 1200000.00),
                ('R$ 500.00', 500.00),
            ],
            'numbers': [
                ('12345', 12345),
                ('000123', 123),
            ],
        }
        
        results_by_type = {}
        
        for data_type, cases in test_cases.items():
            correct = 0
            accuracy = 0
            
            logger.info(f"\nTesting {data_type}:")
            
            for input_val, expected in cases:
                if data_type == 'dates':
                    result = normalizer.normalize_date(input_val)
                elif data_type == 'amounts':
                    result = normalizer.normalize_amount(input_val)
                else:
                    result = normalizer.normalize_numeric(input_val)
                
                is_correct = result == expected
                if is_correct:
                    correct += 1
                
                logger.info(
                    f"  {input_val} -> {result} "
                    f"(expected {expected}) {'✓' if is_correct else '✗'}"
                )
            
            accuracy = correct / len(cases) if cases else 0
            logger.info(f"  Accuracy: {accuracy*100:.1f}% ({correct}/{len(cases)})")
            
            results_by_type[data_type] = accuracy
        
        self.results['normalization_accuracy'] = results_by_type


def main():
    """Main entry point"""
    
    annotation_dir = Path('faturas-images/annotations')
    output_dir = 'semantic_normalization_results'
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run comprehensive test suite
    suite = ComprehensiveSemanticTestSuite(output_dir=output_dir)
    suite.run_all_tests(str(annotation_dir))
    
    logger.info("\n" + "="*80)
    logger.info("SEMANTIC NORMALIZATION TESTING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
