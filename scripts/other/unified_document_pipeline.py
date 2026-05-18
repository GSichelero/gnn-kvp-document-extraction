#!/usr/bin/env python3
"""
Unified Document Understanding Pipeline
========================================

Complete pipeline for optional local document processing:
1. OCR - Text detection and recognition (Tesseract/EasyOCR)
2. Layout Detection - Detect document regions (YOLOv8)
3. Table Detection - Find tables in the document (YOLOv8/TATR)
4. Table Structure Recognition - Detect rows, columns, headers (TATR)
5. KVP Extraction - Entity linking for key-value pairs (BROS)
6. Locator Classification - Classify extracted fields (Sentence Transformers)

Usage:
    python unified_document_pipeline.py --image path/to/image.png
    python unified_document_pipeline.py --image path/to/image.png --output results/
    python unified_document_pipeline.py --pdf path/to/document.pdf --pages 0,1,2
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add repository scripts and project root to path. DOCUMENT_KVP_PROJECT_ROOT
# can override the default when external local data/modules are needed.
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
CONTRIB_ROOT = SCRIPTS_DIR.parent
PROJECT_ROOT = Path(os.environ.get("DOCUMENT_KVP_PROJECT_ROOT", CONTRIB_ROOT)).resolve()
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BoundingBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class TextBlock:
    """OCR text block"""
    text: str
    bbox: BoundingBox
    confidence: float


@dataclass
class TableRegion:
    """Detected table region"""
    bbox: BoundingBox
    confidence: float
    rows: List[BoundingBox] = field(default_factory=list)
    columns: List[BoundingBox] = field(default_factory=list)
    cells: List[Dict] = field(default_factory=list)


@dataclass
class KeyValuePair:
    """Extracted key-value pair"""
    key: str
    value: str
    key_bbox: Optional[BoundingBox] = None
    value_bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    category: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    image_path: str
    processing_time: float
    ocr_results: List[TextBlock] = field(default_factory=list)
    tables: List[TableRegion] = field(default_factory=list)
    key_value_pairs: List[KeyValuePair] = field(default_factory=list)
    classified_fields: Dict[str, List[KeyValuePair]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OCREngine:
    """OCR Engine supporting multiple backends"""

    def __init__(self, backend: str = "easyocr", lang: List[str] = ["pt", "en"]):
        self.backend = backend
        self.lang = lang
        self.engine = None
        self._load_engine()

    def _load_engine(self):
        """Load the OCR engine"""
        if self.backend == "easyocr":
            try:
                import easyocr
                logger.info("Loading EasyOCR...")
                self.engine = easyocr.Reader(self.lang, gpu=True)
                logger.info("EasyOCR loaded")
            except ImportError:
                logger.warning("EasyOCR not available, falling back to Tesseract")
                self.backend = "tesseract"
                self._load_tesseract()
        else:
            self._load_tesseract()

    def _load_tesseract(self):
        """Load Tesseract OCR"""
        try:
            import pytesseract
            self.engine = pytesseract
            logger.info("Tesseract OCR loaded")
        except ImportError:
            raise ImportError("Neither EasyOCR nor Tesseract available. Install with: pip install easyocr or pip install pytesseract")

    def extract_text(self, image) -> List[TextBlock]:
        """Extract text from image"""
        import numpy as np
        from PIL import Image

        # Convert to numpy if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)

        text_blocks = []

        if self.backend == "easyocr":
            results = self.engine.readtext(image)
            for bbox, text, conf in results:
                # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                text_blocks.append(TextBlock(
                    text=text,
                    bbox=BoundingBox(min(x_coords), min(y_coords), max(x_coords), max(y_coords)),
                    confidence=conf
                ))
        else:
            # Tesseract
            data = self.engine.image_to_data(image, output_type=self.engine.Output.DICT, lang='por+eng')
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    text_blocks.append(TextBlock(
                        text=data['text'][i],
                        bbox=BoundingBox(
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ),
                        confidence=data['conf'][i] / 100.0
                    ))

        return text_blocks


class TableDetector:
    """Table detection using YOLO or Table Transformer"""

    def __init__(self, model_type: str = "yolo", model_path: Optional[str] = None):
        self.model_type = model_type
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the table detection model"""
        if self.model_type == "yolo":
            try:
                from ultralytics import YOLO

                # Try to find trained YOLO table detection model
                trained_model = PROJECT_ROOT / "runs" / "detect" / "yolo_table_detection" / "train_20260122_233742" / "weights" / "best.pt"

                if self.model_path and Path(self.model_path).exists():
                    model_file = self.model_path
                elif trained_model.exists():
                    model_file = str(trained_model)
                    logger.info("Using trained table detection model")
                elif Path(PROJECT_ROOT / "yolov8s.pt").exists():
                    model_file = str(PROJECT_ROOT / "yolov8s.pt")
                else:
                    model_file = "yolov8n.pt"  # Will download

                logger.info(f"Loading YOLO model: {model_file}")
                self.model = YOLO(model_file)
                logger.info("YOLO loaded for table detection")
            except ImportError:
                logger.warning("YOLO not available, using Table Transformer")
                self.model_type = "tatr"
                self._load_tatr()
        else:
            self._load_tatr()

    def _load_tatr(self):
        """Load Table Transformer for detection"""
        try:
            from transformers import TableTransformerForObjectDetection, DetrImageProcessor

            logger.info("Loading Table Transformer for detection...")
            self.processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            logger.info("Table Transformer (detection) loaded")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def detect(self, image, conf_threshold: float = 0.25) -> List[TableRegion]:
        """Detect tables in image"""
        import numpy as np
        from PIL import Image
        import torch

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        tables = []

        if self.model_type == "yolo":
            results = self.model(image, verbose=False, conf=conf_threshold)
            for r in results:
                for box in r.boxes:
                    # Get class and confidence
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()

                    # Accept all classes from trained table detector
                    # Class 0 and 1 are typically table types
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Log detection for debugging
                    logger.debug(f"Detected class {cls_id} with conf {conf:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

                    tables.append(TableRegion(
                        bbox=BoundingBox(x1, y1, x2, y2),
                        confidence=conf
                    ))
        else:
            # Table Transformer
            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=0.7, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                tables.append(TableRegion(
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=score.item()
                ))

        return tables


class TableStructureRecognizer:
    """Table structure recognition using Table Transformer"""

    def __init__(self):
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load Table Transformer for structure recognition"""
        try:
            from transformers import TableTransformerForObjectDetection, DetrImageProcessor
            import torch

            logger.info("Loading Table Transformer for structure recognition...")
            self.processor = DetrImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Table Structure Recognizer loaded on {self.device}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def recognize_structure(self, image, table_region: TableRegion) -> TableRegion:
        """Recognize structure within a table region"""
        import torch
        from PIL import Image
        import numpy as np

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Crop table region
        bbox = table_region.bbox
        table_image = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))

        # Process
        inputs = self.processor(images=table_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([table_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        # Structure classes: 0=table, 1=column, 2=row, 3=column header, 4=projected row header, 5=spanning cell
        structure_labels = {
            0: "table",
            1: "column",
            2: "row",
            3: "column_header",
            4: "row_header",
            5: "spanning_cell"
        }

        rows = []
        columns = []
        cells = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_id = label.item()
            x1, y1, x2, y2 = box.tolist()

            # Adjust coordinates to full image
            adj_bbox = BoundingBox(
                bbox.x1 + x1, bbox.y1 + y1,
                bbox.x1 + x2, bbox.y1 + y2
            )

            if label_id == 2:  # row
                rows.append(adj_bbox)
            elif label_id == 1:  # column
                columns.append(adj_bbox)
            else:
                cells.append({
                    "type": structure_labels.get(label_id, "unknown"),
                    "bbox": adj_bbox.to_list(),
                    "confidence": score.item()
                })

        table_region.rows = rows
        table_region.columns = columns
        table_region.cells = cells

        return table_region


class KVPExtractor:
    """Key-Value Pair extraction using custom ML model or BROS"""

    def __init__(self, model_name: str = "custom_ml", model_path: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.ml_extractor = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load KVP extraction model"""
        if self.model_name == "custom_ml":
            self._load_custom_ml()
        else:
            self._load_bros()

    def _load_custom_ml(self):
        """Load custom ML KVP model"""
        try:
            # Import the custom ML extractor
            import sys
            import torch
            sys.path.insert(0, str(PROJECT_ROOT))
            from kv_extractor_ml import MLKVExtractor

            # Find the trained model
            if self.model_path and Path(self.model_path).exists():
                model_file = self.model_path
            elif (PROJECT_ROOT / "kv_model_best.pt").exists():
                model_file = str(PROJECT_ROOT / "kv_model_best.pt")
            else:
                # Fallback to BROS
                logger.warning("Custom ML model not found, falling back to BROS")
                self._load_bros()
                return

            logger.info(f"Loading custom ML KVP model: {model_file}")
            self.ml_extractor = MLKVExtractor(model_path=model_file)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Custom ML KVP extractor loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load custom ML model: {e}")
            logger.info("Falling back to BROS model")
            self._load_bros()

    def _load_bros(self):
        """Load BROS model as fallback"""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            logger.info("Loading BROS model for KVP extraction...")
            model_id = "jinho8345/bros-base-uncased"

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"BROS loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load BROS model: {e}")
            logger.info("Using rule-based KVP extraction as fallback")
            self.model = None

    def extract_pairs(self, text_blocks: List[TextBlock], image_size: Tuple[int, int]) -> List[KeyValuePair]:
        """Extract key-value pairs from text blocks"""
        pairs = []

        # Use custom ML extractor if available
        if self.ml_extractor is not None:
            pairs = self._extract_with_custom_ml(text_blocks, image_size)
        elif self.model is not None:
            pairs = self._extract_with_model(text_blocks, image_size)

        # Always run rule-based as supplement
        rule_pairs = self._extract_rule_based(text_blocks)

        # Merge, avoiding duplicates
        existing_keys = {p.key.lower() for p in pairs}
        for rp in rule_pairs:
            if rp.key.lower() not in existing_keys:
                pairs.append(rp)

        return pairs

    def _extract_with_custom_ml(self, text_blocks: List[TextBlock], image_size: Tuple[int, int]) -> List[KeyValuePair]:
        """Extract using custom ML model"""
        from kv_extractor_ml import TextEntity

        # Convert TextBlocks to TextEntities
        entities = []
        for i, tb in enumerate(text_blocks):
            entity = TextEntity(
                id=str(i),
                text=tb.text,
                box=(int(tb.bbox.x1), int(tb.bbox.y1), int(tb.bbox.x2), int(tb.bbox.y2)),
                confidence=tb.confidence
            )
            entities.append(entity)

        # Extract pairs using ML model
        ml_pairs = self.ml_extractor.extract_from_entities(entities)

        # Convert to KeyValuePair format
        pairs = []
        for ml_pair in ml_pairs:
            pairs.append(KeyValuePair(
                key=ml_pair.key.text,
                value=ml_pair.value.text,
                key_bbox=BoundingBox(
                    ml_pair.key.x1, ml_pair.key.y1,
                    ml_pair.key.x2, ml_pair.key.y2
                ),
                value_bbox=BoundingBox(
                    ml_pair.value.x1, ml_pair.value.y1,
                    ml_pair.value.x2, ml_pair.value.y2
                ),
                confidence=ml_pair.confidence
            ))

        return pairs

    def _extract_with_model(self, text_blocks: List[TextBlock], image_size: Tuple[int, int]) -> List[KeyValuePair]:
        """Extract using BROS model"""
        import torch

        # Prepare input
        texts = [tb.text for tb in text_blocks]
        if not texts:
            return []

        # Simple approach: encode and look for patterns
        # Full BROS requires proper entity linking training
        # This is a simplified version
        pairs = []

        # Use proximity-based linking
        for i, block in enumerate(text_blocks):
            text = block.text.strip()

            # Check if this looks like a key (ends with : or similar)
            if text.endswith(':') or text.endswith('='):
                key = text.rstrip(':=').strip()

                # Find nearest text block to the right or below
                best_value = None
                best_dist = float('inf')

                for j, other in enumerate(text_blocks):
                    if i == j:
                        continue

                    # Check if to the right or below
                    if other.bbox.x1 >= block.bbox.x1 or other.bbox.y1 >= block.bbox.y2:
                        dist = self._distance(block.bbox, other.bbox)
                        if dist < best_dist:
                            best_dist = dist
                            best_value = other

                if best_value and best_dist < 200:  # Max distance threshold
                    pairs.append(KeyValuePair(
                        key=key,
                        value=best_value.text,
                        key_bbox=block.bbox,
                        value_bbox=best_value.bbox,
                        confidence=min(block.confidence, best_value.confidence)
                    ))

        return pairs

    def _extract_rule_based(self, text_blocks: List[TextBlock]) -> List[KeyValuePair]:
        """Generic proximity fallback when no trained KVP model is available."""
        pairs = []

        for i, block in enumerate(text_blocks):
            key_text = block.text.strip()
            if not key_text.endswith((':', '=')):
                continue

            best_value = None
            best_dist = float('inf')
            for j, other in enumerate(text_blocks):
                if i == j:
                    continue

                y_overlap = min(block.bbox.y2, other.bbox.y2) - max(block.bbox.y1, other.bbox.y1)
                same_line = y_overlap > min(block.bbox.height, other.bbox.height) * 0.5
                to_right = other.bbox.x1 > block.bbox.x2
                below = other.bbox.y1 >= block.bbox.y2

                if same_line and to_right:
                    dist = other.bbox.x1 - block.bbox.x2
                elif below and abs(other.bbox.center[0] - block.bbox.center[0]) <= max(block.bbox.width, other.bbox.width):
                    dist = other.bbox.y1 - block.bbox.y2
                else:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_value = other

            if best_value and best_dist < 200:
                pairs.append(KeyValuePair(
                    key=key_text.rstrip(':=').strip(),
                    value=best_value.text,
                    key_bbox=block.bbox,
                    value_bbox=best_value.bbox,
                    confidence=min(block.confidence, best_value.confidence)
                ))

        return pairs

    def _distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between two bounding boxes"""
        import math
        c1 = bbox1.center
        c2 = bbox2.center
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


class FieldClassifier:
    """Classify extracted fields using sentence transformers"""

    def __init__(self):
        self.classifier = None
        self._load_classifier()

    def _load_classifier(self):
        """Load the locator classifier"""
        try:
            from locator_classifier import LocatorClassifier
            logger.info("Loading Locator Classifier...")
            self.classifier = LocatorClassifier()
            logger.info("Locator Classifier loaded")
        except ImportError as e:
            logger.warning(f"Could not load LocatorClassifier: {e}")
            self.classifier = None

    def classify(self, pairs: List[KeyValuePair]) -> Dict[str, List[KeyValuePair]]:
        """Classify key-value pairs into categories"""
        classified = {}

        if self.classifier is None:
            # Simple rule-based classification
            return self._classify_rule_based(pairs)

        # Use sentence transformer classifier
        for pair in pairs:
            # Create pseudo-locator for classification
            locator_text = f"{pair.key}: {pair.value}"

            # Get embedding and classify
            embedding = self.classifier.model.encode([locator_text])[0]

            # Find best matching category
            best_category = None
            best_score = 0

            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            for category, cat_embedding in self.classifier.category_embeddings.items():
                score = cosine_similarity([embedding], [cat_embedding])[0][0]
                if score > best_score:
                    best_score = score
                    best_category = category

            pair.category = best_category

            if best_category not in classified:
                classified[best_category] = []
            classified[best_category].append(pair)

        return classified

    def _classify_rule_based(self, pairs: List[KeyValuePair]) -> Dict[str, List[KeyValuePair]]:
        """Classification fallback without domain-specific hardcoded categories."""
        classified = {}

        for pair in pairs:
            pair.category = "unclassified"
            classified.setdefault(pair.category, []).append(pair)

        return classified

class UnifiedDocumentPipeline:
    """
    Main pipeline orchestrating all document understanding components.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the pipeline.

        Args:
            config: Configuration dictionary with optional settings
        """
        self.config = config or {}

        # Initialize components
        logger.info("=" * 60)
        logger.info("INITIALIZING UNIFIED DOCUMENT PIPELINE")
        logger.info("=" * 60)

        self.ocr = OCREngine(
            backend=self.config.get("ocr_backend", "easyocr"),
            lang=self.config.get("ocr_lang", ["pt", "en"])
        )

        self.table_detector = TableDetector(
            model_type=self.config.get("table_detector", "yolo"),
            model_path=self.config.get("table_model_path")
        )

        self.table_structure = TableStructureRecognizer()

        self.kvp_extractor = KVPExtractor(
            model_name=self.config.get("kvp_model", "custom_ml")
        )

        self.field_classifier = FieldClassifier()

        logger.info("=" * 60)
        logger.info("PIPELINE INITIALIZED")
        logger.info("=" * 60)

    def process(self, image_path: str, save_visualization: bool = True) -> PipelineResult:
        """
        Process a document image through the full pipeline.

        Args:
            image_path: Path to the image file
            save_visualization: Whether to save visualization of results

        Returns:
            PipelineResult with all extracted information
        """
        from PIL import Image
        import numpy as np

        start_time = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {image_path}")
        logger.info(f"{'='*60}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_size = image.size

        result = PipelineResult(
            image_path=image_path,
            processing_time=0,
            metadata={
                "image_size": image_size,
                "timestamp": datetime.now().isoformat()
            }
        )

        # Step 1: OCR
        logger.info("\nStep 1: OCR - Extracting text...")
        step_start = time.time()
        result.ocr_results = self.ocr.extract_text(image_np)
        logger.info(f"   Found {len(result.ocr_results)} text blocks ({time.time()-step_start:.2f}s)")

        # Step 2: Table Detection
        logger.info("\nStep 2: Table Detection...")
        step_start = time.time()
        result.tables = self.table_detector.detect(image)
        logger.info(f"   Found {len(result.tables)} tables ({time.time()-step_start:.2f}s)")

        # Step 3: Table Structure Recognition
        logger.info("\nStep 3: Table Structure Recognition...")
        step_start = time.time()
        for i, table in enumerate(result.tables):
            result.tables[i] = self.table_structure.recognize_structure(image, table)
            logger.info(f"   Table {i+1}: {len(table.rows)} rows, {len(table.columns)} columns")
        logger.info(f"   Structure recognition complete ({time.time()-step_start:.2f}s)")

        # Step 4: KVP Extraction
        logger.info("\nStep 4: Key-Value Pair Extraction...")
        step_start = time.time()
        result.key_value_pairs = self.kvp_extractor.extract_pairs(result.ocr_results, image_size)
        logger.info(f"   Extracted {len(result.key_value_pairs)} key-value pairs ({time.time()-step_start:.2f}s)")

        # Step 5: Field Classification
        logger.info("\nStep 5: Field Classification...")
        step_start = time.time()
        result.classified_fields = self.field_classifier.classify(result.key_value_pairs)
        logger.info(f"   Classified into {len(result.classified_fields)} categories ({time.time()-step_start:.2f}s)")

        # Calculate total time
        result.processing_time = time.time() - start_time

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"   Total processing time: {result.processing_time:.2f}s")
        logger.info(f"   Text blocks: {len(result.ocr_results)}")
        logger.info(f"   Tables detected: {len(result.tables)}")
        logger.info(f"   Key-value pairs: {len(result.key_value_pairs)}")
        logger.info(f"   Field categories: {len(result.classified_fields)}")

        # Print extracted KVPs
        if result.key_value_pairs:
            logger.info("\nEXTRACTED KEY-VALUE PAIRS:")
            for kvp in result.key_value_pairs[:10]:  # Show first 10
                logger.info(f"   [{kvp.category}] {kvp.key}: {kvp.value}")
            if len(result.key_value_pairs) > 10:
                logger.info(f"   ... and {len(result.key_value_pairs) - 10} more")

        # Save visualization
        if save_visualization:
            self._save_visualization(image, result, image_path)

        return result

    def _save_visualization(self, image, result: PipelineResult, image_path: str):
        """Save visualization of results"""
        try:
            from PIL import ImageDraw, ImageFont
            import numpy as np

            # Create a copy for drawing
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)

            # Draw tables (red)
            for table in result.tables:
                bbox = table.bbox
                draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline="red", width=3)

                # Draw rows (blue)
                for row in table.rows:
                    draw.rectangle([row.x1, row.y1, row.x2, row.y2], outline="blue", width=1)

                # Draw columns (green)
                for col in table.columns:
                    draw.rectangle([col.x1, col.y1, col.x2, col.y2], outline="green", width=1)

            # Draw KVP bboxes (orange for keys, purple for values)
            for kvp in result.key_value_pairs:
                if kvp.key_bbox:
                    bbox = kvp.key_bbox
                    draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline="orange", width=2)
                if kvp.value_bbox:
                    bbox = kvp.value_bbox
                    draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline="purple", width=2)

            # Save
            output_path = Path(image_path).stem + "_pipeline_result.png"
            vis_image.save(output_path)
            logger.info(f"\nVisualization saved: {output_path}")

        except Exception as e:
            logger.warning(f"Could not save visualization: {e}")

    def save_results(self, result: PipelineResult, output_path: str):
        """Save results to JSON file"""

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        # Convert to serializable format
        output = {
            "image_path": result.image_path,
            "processing_time": result.processing_time,
            "metadata": result.metadata,
            "ocr_results": [
                {
                    "text": tb.text,
                    "bbox": [float(x) for x in tb.bbox.to_list()],
                    "confidence": float(tb.confidence)
                }
                for tb in result.ocr_results
            ],
            "tables": [
                {
                    "bbox": [float(x) for x in t.bbox.to_list()],
                    "confidence": float(t.confidence),
                    "rows": [[float(x) for x in r.to_list()] for r in t.rows],
                    "columns": [[float(x) for x in c.to_list()] for c in t.columns],
                    "cells": convert_to_serializable(t.cells)
                }
                for t in result.tables
            ],
            "key_value_pairs": [
                {
                    "key": kvp.key,
                    "value": kvp.value,
                    "category": kvp.category,
                    "confidence": float(kvp.confidence),
                    "key_bbox": [float(x) for x in kvp.key_bbox.to_list()] if kvp.key_bbox else None,
                    "value_bbox": [float(x) for x in kvp.value_bbox.to_list()] if kvp.value_bbox else None
                }
                for kvp in result.key_value_pairs
            ],
            "classified_fields": {
                category: [
                    {"key": kvp.key, "value": kvp.value, "confidence": float(kvp.confidence)}
                    for kvp in pairs
                ]
                for category, pairs in result.classified_fields.items()
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Document Understanding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single image
    python unified_document_pipeline.py --image document.png

    # Process with custom output directory
    python unified_document_pipeline.py --image document.png --output results/

    # Use Tesseract instead of EasyOCR
    python unified_document_pipeline.py --image document.png --ocr tesseract

    # Process multiple images
    python unified_document_pipeline.py --images_dir data/images/ --output results/
        """
    )

    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--images_dir", type=str, help="Directory with images to process")
    parser.add_argument("--output", type=str, default="pipeline_results", help="Output directory")
    parser.add_argument("--ocr", type=str, default="easyocr", choices=["easyocr", "tesseract"], help="OCR backend")
    parser.add_argument("--table_detector", type=str, default="yolo", choices=["yolo", "tatr"], help="Table detection model")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization output")

    args = parser.parse_args()

    # Validate input
    if not args.image and not args.images_dir:
        parser.error("Either --image or --images_dir must be provided")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Configuration
    config = {
        "ocr_backend": args.ocr,
        "table_detector": args.table_detector,
    }

    # Initialize pipeline
    pipeline = UnifiedDocumentPipeline(config)

    # Process images
    if args.image:
        images = [args.image]
    else:
        images = list(Path(args.images_dir).glob("*.png")) + \
                 list(Path(args.images_dir).glob("*.jpg")) + \
                 list(Path(args.images_dir).glob("*.jpeg"))

    logger.info(f"\nProcessing {len(images)} image(s)...")

    for image_path in images:
        try:
            result = pipeline.process(str(image_path), save_visualization=not args.no_viz)

            # Save results
            output_file = output_dir / f"{Path(image_path).stem}_results.json"
            pipeline.save_results(result, str(output_file))

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\nProcessing complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
