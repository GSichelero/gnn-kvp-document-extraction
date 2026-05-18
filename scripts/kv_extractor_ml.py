#!/usr/bin/env python3
"""
ML-enhanced key-value pair extractor.
Learns entity relationships and key-like text hints from ground-truth annotations.
"""

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import Counter, defaultdict
import pickle

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. ML features disabled.")


@dataclass
class TextEntity:
    """Represents a text entity with bounding box"""
    id: str
    text: str
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 1.0
    label: str = "other"

    @property
    def x1(self): return self.box[0]
    @property
    def y1(self): return self.box[1]
    @property
    def x2(self): return self.box[2]
    @property
    def y2(self): return self.box[3]
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def center_x(self): return (self.x1 + self.x2) / 2
    @property
    def center_y(self): return (self.y1 + self.y2) / 2
    @property
    def area(self): return self.width * self.height


@dataclass
class KVPair:
    """Represents a key-value pair"""
    key: TextEntity
    value: TextEntity
    confidence: float = 1.0
    method: str = "ml"

    def to_dict(self) -> dict:
        return {
            "key": {"text": self.key.text, "box": list(self.key.box), "id": self.key.id},
            "value": {"text": self.value.text, "box": list(self.value.box), "id": self.value.id},
            "confidence": self.confidence,
            "method": self.method
        }


class TextFeatureExtractor:
    """Extracts features from text entities"""

    def __init__(self, max_key_terms: int = 64):
        self.text_vocab: Dict[str, int] = {}
        self.key_terms: List[str] = []
        self.vocab_size = 0
        self.max_text_len = 50
        self.max_key_terms = max_key_terms

    def build_vocab(self, texts: List[str], key_texts: Optional[List[str]] = None):
        """Build vocabulary and key-term hints from training annotations."""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text.lower())
            for word in words:
                word_counts[word] += 1

        self.text_vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
            if count >= 1:
                self.text_vocab[word] = len(self.text_vocab)

        self.vocab_size = len(self.text_vocab)

        key_counts = Counter()
        for text in key_texts or []:
            key_counts.update(self._tokenize(text.lower()))
        self.key_terms = [
            word
            for word, _ in key_counts.most_common(self.max_key_terms)
            if len(word) > 1 and word in self.text_vocab
        ]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\w+', text.lower())

    def looks_like_key(self, entity: TextEntity) -> bool:
        """Generic key-like check using annotation labels and learned terms."""
        text = entity.text.strip()
        if entity.label in {"question", "header"}:
            return True
        if text.endswith((':', '=')):
            return True
        if not self.key_terms:
            return False
        tokens = set(self._tokenize(text.lower()))
        return any(term in tokens for term in self.key_terms)

    def extract_text_features(self, entity: TextEntity) -> np.ndarray:
        """Extract text-based features"""
        text = entity.text.strip()

        features = []

        # 1. Length features
        features.append(len(text) / 50.0)  # Normalized length
        features.append(len(text.split()) / 10.0)  # Word count

        # 2. Character composition
        num_digits = sum(c.isdigit() for c in text)
        num_alpha = sum(c.isalpha() for c in text)
        num_special = len(text) - num_digits - num_alpha

        total = max(len(text), 1)
        features.append(num_digits / total)  # Digit ratio
        features.append(num_alpha / total)   # Alpha ratio
        features.append(num_special / total)  # Special char ratio

        # 3. Pattern features
        features.append(1.0 if text.endswith(':') else 0.0)  # Ends with colon
        features.append(1.0 if re.match(r'^\d+[.,]\d+$', text) else 0.0)  # Number
        features.append(1.0 if re.match(r'^\d{2}/\d{2}/\d{4}$', text) else 0.0)  # Date
        features.append(1.0 if re.match(r'^\d{2}/\d{4}$', text) else 0.0)  # Month/Year
        features.append(1.0 if 'R$' in text or re.match(r'^\d+[.,]\d{2}$', text) else 0.0)  # Currency

        # 4. Training-derived key-term matching
        text_lower = text.lower()
        tokens = set(self._tokenize(text_lower))
        if self.key_terms:
            keyword_match = sum(1 for kw in self.key_terms if kw in tokens)
            denom = max(1, min(3, len(self.key_terms)))
            features.append(min(keyword_match / denom, 1.0))
        else:
            features.append(0.0)

        # 5. Case features
        features.append(1.0 if text.isupper() else 0.0)
        features.append(1.0 if (text and text[0].isupper()) else 0.0)

        return np.array(features, dtype=np.float32)

    def extract_spatial_features(self, entity: TextEntity,
                                  page_width: float = 1600,
                                  page_height: float = 2000) -> np.ndarray:
        """Extract spatial features"""
        features = []

        # Normalized position
        features.append(entity.x1 / page_width)
        features.append(entity.y1 / page_height)
        features.append(entity.x2 / page_width)
        features.append(entity.y2 / page_height)

        # Center position
        features.append(entity.center_x / page_width)
        features.append(entity.center_y / page_height)

        # Size features
        features.append(entity.width / page_width)
        features.append(entity.height / page_height)
        features.append(entity.area / (page_width * page_height))

        # Aspect ratio
        features.append(entity.width / max(entity.height, 1))

        return np.array(features, dtype=np.float32)

    def extract_pair_features(self, key: TextEntity, value: TextEntity,
                               page_width: float = 1600,
                               page_height: float = 2000) -> np.ndarray:
        """Extract features for a key-value pair candidate"""
        features = []

        # Relative position
        dx = (value.center_x - key.center_x) / page_width
        dy = (value.center_y - key.center_y) / page_height
        features.append(dx)
        features.append(dy)

        # Distance
        dist = np.sqrt(dx**2 + dy**2)
        features.append(dist)

        # Horizontal gap
        gap = (value.x1 - key.x2) / page_width if value.x1 > key.x2 else 0
        features.append(gap)

        # Vertical alignment (overlap ratio)
        overlap_y = min(key.y2, value.y2) - max(key.y1, value.y1)
        min_height = min(key.height, value.height)
        features.append(overlap_y / max(min_height, 1))

        # Size ratio
        features.append(value.width / max(key.width, 1))
        features.append(value.height / max(key.height, 1))

        # Is value to the right of key?
        features.append(1.0 if value.x1 > key.x2 else 0.0)

        # Is value below key?
        features.append(1.0 if value.y1 > key.y2 else 0.0)

        # Are they on the same line?
        same_line = overlap_y > min_height * 0.5 if min_height > 0 else False
        features.append(1.0 if same_line else 0.0)

        return np.array(features, dtype=np.float32)


if TORCH_AVAILABLE:
    class KVClassifier(nn.Module):
        """Neural network to classify key-value pair candidates"""

        def __init__(self, text_feature_dim: int = 13,
                     spatial_feature_dim: int = 10,
                     pair_feature_dim: int = 10,
                     hidden_dim: int = 64):
            super().__init__()

            # Entity encoders
            entity_dim = text_feature_dim + spatial_feature_dim  # 13 + 10 = 23
            self.entity_encoder = nn.Sequential(
                nn.Linear(entity_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )

            # Pair classifier
            pair_input_dim = hidden_dim + pair_feature_dim  # key + value + pair features
            self.pair_classifier = nn.Sequential(
                nn.Linear(pair_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        def encode_entity(self, text_features: torch.Tensor,
                          spatial_features: torch.Tensor) -> torch.Tensor:
            """Encode a single entity"""
            x = torch.cat([text_features, spatial_features], dim=-1)
            return self.entity_encoder(x)

        def forward(self, key_text: torch.Tensor, key_spatial: torch.Tensor,
                    value_text: torch.Tensor, value_spatial: torch.Tensor,
                    pair_features: torch.Tensor) -> torch.Tensor:
            """Predict probability of being a valid key-value pair"""
            key_enc = self.encode_entity(key_text, key_spatial)
            value_enc = self.encode_entity(value_text, value_spatial)

            combined = torch.cat([key_enc, value_enc, pair_features], dim=-1)
            return self.pair_classifier(combined).squeeze(-1)


    class KVDataset(Dataset):
        """Dataset for training KV classifier"""

        def __init__(self, samples: List[dict]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            return {
                'key_text': torch.tensor(s['key_text_features']),
                'key_spatial': torch.tensor(s['key_spatial_features']),
                'value_text': torch.tensor(s['value_text_features']),
                'value_spatial': torch.tensor(s['value_spatial_features']),
                'pair_features': torch.tensor(s['pair_features']),
                'label': torch.tensor(s['label'], dtype=torch.float32)
            }


class MLKVExtractor:
    """ML-enhanced key-value pair extractor"""

    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = TextFeatureExtractor()
        self.model: Optional['KVClassifier'] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None

        if model_path and TORCH_AVAILABLE:
            self.load_model(model_path)

    def load_annotations(self, annotation_dir: Path) -> List[dict]:
        """Load annotations for training"""
        annotation_dir = Path(annotation_dir)
        all_data = []

        for json_file in annotation_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file)
                    all_data.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        return all_data

    @staticmethod
    def _iter_link_targets(item: dict):
        """Yield linked target ids from common FUNSD-style link encodings."""
        source_id = item.get('id')
        for link in item.get('linking', []):
            if isinstance(link, (list, tuple)):
                if len(link) == 2:
                    if link[0] == source_id:
                        yield link[1]
                    elif source_id is None:
                        yield link[1]
                elif len(link) == 1:
                    yield link[0]
            else:
                yield link

    def prepare_training_data(self, annotations: List[dict]) -> List[dict]:
        """Prepare training samples from annotations"""
        all_texts = []
        key_texts = []
        for data in annotations:
            form = data.get('form', [])
            entities = {item.get('id'): item for item in form}
            linked_sources = set()
            for item in form:
                all_texts.append(item.get('text', ''))
                for linked_id in self._iter_link_targets(item):
                    if linked_id in entities:
                        linked_sources.add(item.get('id'))
            for item in form:
                if item.get('id') in linked_sources or item.get('label') in {"question", "header"}:
                    key_texts.append(item.get('text', ''))
        self.feature_extractor.build_vocab(all_texts, key_texts=key_texts)

        samples = []

        for data in annotations:
            entities = {item['id']: item for item in data.get('form', [])}

            # Get positive pairs (ground truth links)
            positive_pairs = set()
            for item in data.get('form', []):
                for linked_id in self._iter_link_targets(item):
                    if linked_id in entities:
                        positive_pairs.add((item['id'], linked_id))

            if not positive_pairs:
                continue

            # Create samples
            entity_list = list(data.get('form', []))

            for i, key_item in enumerate(entity_list):
                key_entity = self._item_to_entity(key_item)
                key_text_feat = self.feature_extractor.extract_text_features(key_entity)
                key_spatial_feat = self.feature_extractor.extract_spatial_features(key_entity)

                # Sample candidates
                for j, value_item in enumerate(entity_list):
                    if i == j:
                        continue

                    value_entity = self._item_to_entity(value_item)

                    # Skip if too far (basic filter)
                    if abs(value_entity.center_y - key_entity.center_y) > 100 and \
                       value_entity.x1 < key_entity.x1:
                        continue

                    value_text_feat = self.feature_extractor.extract_text_features(value_entity)
                    value_spatial_feat = self.feature_extractor.extract_spatial_features(value_entity)
                    pair_feat = self.feature_extractor.extract_pair_features(key_entity, value_entity)

                    is_positive = (key_item['id'], value_item['id']) in positive_pairs

                    # Downsample negatives
                    if not is_positive and np.random.random() > 0.1:
                        continue

                    samples.append({
                        'key_text_features': key_text_feat,
                        'key_spatial_features': key_spatial_feat,
                        'value_text_features': value_text_feat,
                        'value_spatial_features': value_spatial_feat,
                        'pair_features': pair_feat,
                        'label': 1.0 if is_positive else 0.0
                    })

        return samples

    def _item_to_entity(self, item: dict) -> TextEntity:
        """Convert annotation item to TextEntity"""
        return TextEntity(
            id=item['id'],
            text=item['text'],
            box=tuple(item['box']),
            label=item.get('label', 'other')
        )

    def train(self, annotation_dir: str, epochs: int = 50, lr: float = 0.001):
        """Train the ML model"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Cannot train model.")
            return

        print("Loading annotations...")
        annotations = self.load_annotations(Path(annotation_dir))
        print(f"Loaded {len(annotations)} annotation files")

        print("Preparing training data...")
        samples = self.prepare_training_data(annotations)

        # Count positives and negatives
        n_positive = sum(1 for s in samples if s['label'] > 0.5)
        n_negative = len(samples) - n_positive
        print(f"Training samples: {len(samples)} ({n_positive} positive, {n_negative} negative)")

        if len(samples) < 10:
            print("Not enough training samples!")
            return

        # Split data
        np.random.shuffle(samples)
        split_idx = int(len(samples) * 0.8)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        train_dataset = KVDataset(train_samples)
        val_dataset = KVDataset(val_samples)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Initialize model
        self.model = KVClassifier()
        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Class weights for imbalanced data
        pos_weight = n_negative / max(n_positive, 1)

        best_f1 = 0

        print(f"\nTraining on {self.device}...")
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                pred = self.model(
                    batch['key_text'].to(self.device),
                    batch['key_spatial'].to(self.device),
                    batch['value_text'].to(self.device),
                    batch['value_spatial'].to(self.device),
                    batch['pair_features'].to(self.device)
                )

                labels = batch['label'].to(self.device)

                # Weighted loss
                weight = torch.where(labels > 0.5,
                                    torch.full_like(labels, pos_weight),
                                    torch.ones_like(labels))
                loss = F.binary_cross_entropy(pred, labels, weight=weight)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    pred = self.model(
                        batch['key_text'].to(self.device),
                        batch['key_spatial'].to(self.device),
                        batch['value_text'].to(self.device),
                        batch['value_spatial'].to(self.device),
                        batch['pair_features'].to(self.device)
                    )
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch['label'].numpy())

            # Calculate metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            pred_binary = (all_preds > 0.5).astype(float)
            tp = np.sum((pred_binary == 1) & (all_labels == 1))
            fp = np.sum((pred_binary == 1) & (all_labels == 0))
            fn = np.sum((pred_binary == 0) & (all_labels == 1))

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)

            if (epoch + 1) % 10 == 0 or f1 > best_f1:
                print(f"Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, "
                      f"P={precision:.2%}, R={recall:.2%}, F1={f1:.2%}")

            if f1 > best_f1:
                best_f1 = f1
                self.save_model("kv_model_best.pt")

        print(f"\nTraining complete. Best F1: {best_f1:.2%}")

    def save_model(self, path: str):
        """Save model to file"""
        if self.model is None:
            return

        state = {
            'model_state': self.model.state_dict(),
            'vocab': self.feature_extractor.text_vocab,
            'vocab_size': self.feature_extractor.vocab_size,
            'key_terms': self.feature_extractor.key_terms,
        }
        torch.save(state, path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str):
        """Load model from file"""
        if not TORCH_AVAILABLE:
            return

        state = torch.load(path, map_location=self.device)

        self.feature_extractor.text_vocab = state['vocab']
        self.feature_extractor.vocab_size = state['vocab_size']
        self.feature_extractor.key_terms = state.get('key_terms', [])

        self.model = KVClassifier()
        self.model.load_state_dict(state['model_state'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from: {path}")

    def extract_from_annotation(self, annotation_path: str) -> Tuple[List[TextEntity], List[KVPair], List[KVPair]]:
        """Extract key-value pairs from annotation file"""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entities = [self._item_to_entity(item) for item in data.get('form', [])]

        # Get ground truth pairs
        gt_pairs = self._extract_ground_truth(data)

        # Predict pairs
        if self.model is not None:
            predicted_pairs = self._predict_pairs(entities)
        else:
            predicted_pairs = self._heuristic_pairs(entities)

        return entities, predicted_pairs, gt_pairs

    def _extract_ground_truth(self, data: dict) -> List[KVPair]:
        """Extract ground truth pairs from annotations"""
        entities = {item['id']: item for item in data.get('form', [])}
        pairs = []

        for item in data.get('form', []):
            for linked_id in self._iter_link_targets(item):
                if linked_id in entities:
                    key_entity = self._item_to_entity(item)
                    value_entity = self._item_to_entity(entities[linked_id])
                    pairs.append(KVPair(key_entity, value_entity, 1.0, "ground_truth"))

        return pairs

    def _predict_pairs(self, entities: List[TextEntity]) -> List[KVPair]:
        """Predict pairs using the trained model"""
        if self.model is None or not entities:
            return []

        self.model.eval()
        pairs = []

        with torch.no_grad():
            for key_entity in entities:
                key_text = self.feature_extractor.extract_text_features(key_entity)
                key_spatial = self.feature_extractor.extract_spatial_features(key_entity)

                # Score all potential values
                candidates = []
                for value_entity in entities:
                    if value_entity.id == key_entity.id:
                        continue

                    # Basic filters
                    if value_entity.x1 < key_entity.x1 - 50:
                        continue
                    if abs(value_entity.center_y - key_entity.center_y) > 150:
                        continue

                    value_text = self.feature_extractor.extract_text_features(value_entity)
                    value_spatial = self.feature_extractor.extract_spatial_features(value_entity)
                    pair_feat = self.feature_extractor.extract_pair_features(key_entity, value_entity)

                    # Predict
                    pred = self.model(
                        torch.tensor(key_text).unsqueeze(0).to(self.device),
                        torch.tensor(key_spatial).unsqueeze(0).to(self.device),
                        torch.tensor(value_text).unsqueeze(0).to(self.device),
                        torch.tensor(value_spatial).unsqueeze(0).to(self.device),
                        torch.tensor(pair_feat).unsqueeze(0).to(self.device)
                    )

                    score = pred.item()
                    if score > 0.5:
                        candidates.append((value_entity, score))

                # Take best candidate
                if candidates:
                    candidates.sort(key=lambda x: -x[1])
                    best_value, best_score = candidates[0]
                    pairs.append(KVPair(key_entity, best_value, best_score, "ml"))

        # Remove duplicates (same value used multiple times)
        seen_values = set()
        unique_pairs = []
        for pair in sorted(pairs, key=lambda p: -p.confidence):
            if pair.value.id not in seen_values:
                unique_pairs.append(pair)
                seen_values.add(pair.value.id)

        return unique_pairs

    def _heuristic_pairs(self, entities: List[TextEntity]) -> List[KVPair]:
        """Fallback generic extraction when no trained model is available."""
        pairs = []

        for key_entity in entities:
            if not self.feature_extractor.looks_like_key(key_entity):
                continue

            best_value = None
            best_dist = float('inf')

            for value_entity in entities:
                if value_entity.id == key_entity.id:
                    continue

                y_overlap = min(key_entity.y2, value_entity.y2) - max(key_entity.y1, value_entity.y1)
                same_line = y_overlap > min(key_entity.height, value_entity.height) * 0.5
                to_right = value_entity.x1 > key_entity.x2
                below = value_entity.y1 >= key_entity.y2

                if same_line and to_right:
                    dist = value_entity.x1 - key_entity.x2
                elif below and abs(value_entity.center_x - key_entity.center_x) <= max(key_entity.width, value_entity.width):
                    dist = value_entity.y1 - key_entity.y2
                else:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_value = value_entity

            if best_value:
                pairs.append(KVPair(key_entity, best_value, 0.7, "heuristic"))

        return pairs

    def extract_from_entities(self, entities: List[TextEntity]) -> List[KVPair]:
        """Extract key-value pairs from a list of TextEntity objects.

        This is the main method for integration with other pipelines.

        Args:
            entities: List of TextEntity objects with text and bounding boxes

        Returns:
            List of KVPair objects representing key-value relationships
        """
        if not entities:
            return []

        # Use ML model if available, otherwise fallback to heuristics
        if self.model is not None:
            return self._predict_pairs(entities)
        else:
            return self._heuristic_pairs(entities)

    def evaluate(self, predicted: List[KVPair], ground_truth: List[KVPair]) -> Dict:
        """Evaluate predictions against ground truth"""
        pred_set = {(p.key.text.strip().lower(), p.value.text.strip().lower()) for p in predicted}
        gt_set = {(p.key.text.strip().lower(), p.value.text.strip().lower()) for p in ground_truth}

        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gt_set) if gt_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1,
                "true_positives": tp, "predicted_count": len(pred_set),
                "ground_truth_count": len(gt_set)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ML-enhanced KV pair extraction")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument(
        "--annotation-dir",
        "-a",
        default=os.environ.get("KVP_ANNOTATION_DIR", "data/annotations"),
        help="Directory with FUNSD-style training annotations",
    )
    parser.add_argument("--model", "-m", default="kv_model_best.pt", help="Model path")
    parser.add_argument("--test", "-t", help="Test on specific annotation file")
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    extractor = MLKVExtractor()

    if args.train:
        extractor.train(args.annotation_dir, epochs=args.epochs)

    elif args.test:
        if Path(args.model).exists():
            extractor.load_model(args.model)

        entities, predicted, gt = extractor.extract_from_annotation(args.test)

        print(f"\n{'='*60}")
        print(f"Testing: {args.test}")
        print('='*60)

        print(f"\nPredicted Pairs ({len(predicted)}):")
        for p in predicted[:15]:
            print(f"  [{p.key.text}] -> [{p.value.text}] ({p.confidence:.2f})")

        print(f"\nGround Truth ({len(gt)}):")
        for p in gt:
            print(f"  [{p.key.text}] -> [{p.value.text}]")

        metrics = extractor.evaluate(predicted, gt)
        print(f"\nMetrics: P={metrics['precision']:.2%}, R={metrics['recall']:.2%}, F1={metrics['f1']:.2%}")

    else:
        # Demo: evaluate on all annotations
        annotation_dir = Path(args.annotation_dir)

        if Path(args.model).exists():
            extractor.load_model(args.model)
        else:
            print("No trained model found. Training...")
            extractor.train(args.annotation_dir, epochs=args.epochs)

        total_p, total_r, total_f1 = 0, 0, 0
        count = 0

        for json_file in sorted(annotation_dir.glob("*.json")):
            entities, predicted, gt = extractor.extract_from_annotation(str(json_file))
            if gt:
                metrics = extractor.evaluate(predicted, gt)
                total_p += metrics['precision']
                total_r += metrics['recall']
                total_f1 += metrics['f1']
                count += 1
                print(f"{json_file.name[:40]:<40} P={metrics['precision']:.0%} R={metrics['recall']:.0%} F1={metrics['f1']:.0%}")

        if count > 0:
            print(f"\n{'='*60}")
            print(f"Average: P={total_p/count:.2%}, R={total_r/count:.2%}, F1={total_f1/count:.2%}")


if __name__ == "__main__":
    main()
