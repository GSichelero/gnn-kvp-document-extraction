#!/usr/bin/env python3
"""
Table Detection and Structure Recognition System
Uses Microsoft Table Transformer and other ML models for:
1. Table detection (finding tables in images)
2. Table structure recognition (rows, columns, cells)
3. Cell content extraction with OCR

Supports both online models (Hugging Face) and local YOLO models
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("EasyOCR not available")


@dataclass
class BBox:
    """Bounding box representation"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self): return self.x2 - self.x1
    
    @property
    def height(self): return self.y2 - self.y1
    
    @property
    def center(self): return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self): return self.width * self.height
    
    def to_list(self): return [self.x1, self.y1, self.x2, self.y2]
    
    def iou(self, other: 'BBox') -> float:
        """Calculate IoU with another box"""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = self.area + other.area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


@dataclass
class TableCell:
    """Represents a single table cell"""
    bbox: BBox
    row: int = -1
    col: int = -1
    text: str = ""
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1
    confidence: float = 1.0


@dataclass
class TableRow:
    """Represents a table row"""
    bbox: BBox
    index: int
    cells: List[TableCell] = field(default_factory=list)
    is_header: bool = False


@dataclass 
class TableColumn:
    """Represents a table column"""
    bbox: BBox
    index: int
    cells: List[TableCell] = field(default_factory=list)
    is_header: bool = False


@dataclass
class DetectedTable:
    """Represents a detected table with structure"""
    bbox: BBox
    confidence: float
    rows: List[TableRow] = field(default_factory=list)
    columns: List[TableColumn] = field(default_factory=list)
    cells: List[TableCell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    
    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "rows": [{"bbox": r.bbox.to_list(), "index": r.index, "is_header": r.is_header} 
                     for r in self.rows],
            "columns": [{"bbox": c.bbox.to_list(), "index": c.index} for c in self.columns],
            "cells": [{
                "bbox": c.bbox.to_list(),
                "row": c.row,
                "col": c.col,
                "text": c.text,
                "is_header": c.is_header,
                "row_span": c.row_span,
                "col_span": c.col_span,
                "confidence": c.confidence
            } for c in self.cells]
        }
    
    def to_grid(self) -> List[List[str]]:
        """Convert to 2D grid of text"""
        if self.num_rows == 0 or self.num_cols == 0:
            return []
        
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        
        for cell in self.cells:
            if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text
        
        return grid


class TableTransformerDetector:
    """
    Uses Microsoft's Table Transformer models from Hugging Face:
    - microsoft/table-transformer-detection (for finding tables)
    - microsoft/table-transformer-structure-recognition (for structure)
    """
    
    def __init__(self, device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise ImportError("transformers and torch required")
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load detection model
        logger.info("Loading Table Detection model...")
        self.detection_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self.detection_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        ).to(self.device)
        
        # Load structure recognition model
        logger.info("Loading Table Structure Recognition model...")
        self.structure_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        ).to(self.device)
        
        # Label mappings
        self.detection_labels = {0: "table", 1: "table rotated"}
        self.structure_labels = {
            0: "table", 1: "table column", 2: "table row", 
            3: "table column header", 4: "table projected row header",
            5: "table spanning cell"
        }
        
        logger.info("✓ Models loaded successfully")
    
    def detect_tables(self, image: np.ndarray, threshold: float = 0.7) -> List[DetectedTable]:
        """
        Detect tables in an image
        
        Args:
            image: BGR image array
            threshold: Confidence threshold
            
        Returns:
            List of detected tables with bounding boxes
        """
        # Convert BGR to RGB PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process image
        inputs = self.detection_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run detection
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            tables.append(DetectedTable(
                bbox=BBox(x1, y1, x2, y2),
                confidence=score.item()
            ))
        
        logger.info(f"Detected {len(tables)} tables")
        return tables
    
    def recognize_structure(self, image: np.ndarray, table: DetectedTable, 
                           threshold: float = 0.5) -> DetectedTable:
        """
        Recognize structure of a detected table
        
        Args:
            image: Full image
            table: Detected table with bbox
            threshold: Confidence threshold
            
        Returns:
            Table with structure (rows, columns, cells)
        """
        # Crop table region
        x1, y1, x2, y2 = table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2
        table_crop = image[y1:y2, x1:x2]
        
        # Convert to PIL
        pil_crop = Image.fromarray(cv2.cvtColor(table_crop, cv2.COLOR_BGR2RGB))
        
        # Process
        inputs = self.structure_processor(images=pil_crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run structure recognition
        with torch.no_grad():
            outputs = self.structure_model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([pil_crop.size[::-1]]).to(self.device)
        results = self.structure_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        rows = []
        columns = []
        cells = []
        headers = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.structure_labels.get(label.item(), "unknown")
            bx1, by1, bx2, by2 = [int(v) for v in box.tolist()]
            
            # Adjust coordinates to full image
            bbox = BBox(x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
            
            if label_name == "table row":
                rows.append(TableRow(bbox=bbox, index=len(rows)))
            elif label_name == "table column":
                columns.append(TableColumn(bbox=bbox, index=len(columns)))
            elif label_name == "table column header":
                headers.append(bbox)
        
        # Sort rows and columns
        rows.sort(key=lambda r: r.bbox.y1)
        columns.sort(key=lambda c: c.bbox.x1)
        
        # Update indices after sorting
        for i, row in enumerate(rows):
            row.index = i
        for i, col in enumerate(columns):
            col.index = i
        
        # Generate cells from row/column intersections
        cells = self._generate_cells(rows, columns, headers)
        
        table.rows = rows
        table.columns = columns
        table.cells = cells
        table.num_rows = len(rows)
        table.num_cols = len(columns)
        
        logger.info(f"Structure: {len(rows)} rows, {len(columns)} columns, {len(cells)} cells")
        
        return table
    
    def _generate_cells(self, rows: List[TableRow], columns: List[TableColumn],
                       headers: List[BBox]) -> List[TableCell]:
        """Generate cells from row/column intersections"""
        cells = []
        
        for row in rows:
            for col in columns:
                # Cell is intersection of row and column
                cell_x1 = max(row.bbox.x1, col.bbox.x1)
                cell_y1 = max(row.bbox.y1, col.bbox.y1)
                cell_x2 = min(row.bbox.x2, col.bbox.x2)
                cell_y2 = min(row.bbox.y2, col.bbox.y2)
                
                if cell_x2 > cell_x1 and cell_y2 > cell_y1:
                    cell_bbox = BBox(cell_x1, cell_y1, cell_x2, cell_y2)
                    
                    # Check if cell is in header
                    is_header = any(
                        cell_bbox.iou(h) > 0.5 for h in headers
                    ) or row.index == 0  # First row often header
                    
                    cell = TableCell(
                        bbox=cell_bbox,
                        row=row.index,
                        col=col.index,
                        is_header=is_header
                    )
                    cells.append(cell)
                    
                    # Add to row and column
                    row.cells.append(cell)
                    col.cells.append(cell)
        
        return cells


class YOLOTableDetector:
    """Uses YOLO model for table detection (faster, good for fine-tuned models)"""
    
    def __init__(self, model_path: str = "yolov8s.pt"):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics required")
        
        self.model = YOLO(model_path)
        logger.info(f"Loaded YOLO model: {model_path}")
    
    def detect_tables(self, image: np.ndarray, threshold: float = 0.5) -> List[DetectedTable]:
        """Detect tables using YOLO"""
        results = self.model(image, conf=threshold, verbose=False)
        
        tables = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                tables.append(DetectedTable(
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=float(box.conf[0])
                ))
        
        return tables


class HeuristicStructureRecognizer:
    """
    Heuristic-based table structure recognition
    Uses line detection and cell clustering when ML models aren't available
    Multiple methods available for different table styles
    """
    
    def __init__(self, method: str = "auto"):
        """
        Args:
            method: "lines" (line detection), "ocr_clustering" (cluster OCR boxes),
                   "contours" (find cell contours), "auto" (try all)
        """
        self.method = method
    
    def recognize_structure(self, image: np.ndarray, table: DetectedTable,
                           ocr_boxes: Optional[List] = None) -> DetectedTable:
        """Recognize table structure using image processing"""
        if self.method == "auto":
            # Try methods in order of reliability
            methods = ["lines", "contours", "text_alignment"]
            for method in methods:
                result = self._recognize_with_method(image, table, method, ocr_boxes)
                if result.num_rows > 0 and result.num_cols > 0:
                    return result
            return table
        else:
            return self._recognize_with_method(image, table, self.method, ocr_boxes)
    
    def _recognize_with_method(self, image: np.ndarray, table: DetectedTable, 
                               method: str, ocr_boxes: Optional[List] = None) -> DetectedTable:
        """Apply specific recognition method"""
        if method == "lines":
            return self._recognize_by_lines(image, table)
        elif method == "contours":
            return self._recognize_by_contours(image, table)
        elif method == "text_alignment":
            return self._recognize_by_text_alignment(image, table)
        else:
            return self._recognize_by_lines(image, table)
    
    def _recognize_by_lines(self, image: np.ndarray, table: DetectedTable) -> DetectedTable:
        """Recognize structure by detecting horizontal and vertical lines"""
        x1, y1, x2, y2 = table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2
        table_crop = image[y1:y2, x1:x2]
        
        if table_crop.size == 0:
            return table
        
        # Convert to grayscale
        gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find horizontal and vertical lines with different kernel sizes
        h, w = table_crop.shape[:2]
        
        # Use proportional kernel sizes
        h_kernel_size = max(40, w // 10)
        v_kernel_size = max(40, h // 10)
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        
        # Find line positions
        h_positions = self._find_line_positions(h_lines, horizontal=True, min_gap=15)
        v_positions = self._find_line_positions(v_lines, horizontal=False, min_gap=15)
        
        # Add table edges if not detected
        if not h_positions or h_positions[0] > 10:
            h_positions = [0] + h_positions
        if not h_positions or h_positions[-1] < h - 10:
            h_positions = h_positions + [h]
        if not v_positions or v_positions[0] > 10:
            v_positions = [0] + v_positions
        if not v_positions or v_positions[-1] < w - 10:
            v_positions = v_positions + [w]
        
        return self._create_grid_from_positions(table, h_positions, v_positions, x1, y1)
    
    def _recognize_by_contours(self, image: np.ndarray, table: DetectedTable) -> DetectedTable:
        """Recognize structure by finding cell contours"""
        x1, y1, x2, y2 = table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2
        table_crop = image[y1:y2, x1:x2]
        
        if table_crop.size == 0:
            return table
        
        gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (likely cells)
        min_area = (table_crop.shape[0] * table_crop.shape[1]) // 200
        max_area = (table_crop.shape[0] * table_crop.shape[1]) // 4
        
        cell_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio
                aspect = w / max(h, 1)
                if 0.2 < aspect < 10:
                    cell_boxes.append(BBox(x1 + x, y1 + y, x1 + x + w, y1 + y + h))
        
        if cell_boxes:
            return self._create_grid_from_cells(table, cell_boxes)
        
        return table
    
    def _recognize_by_text_alignment(self, image: np.ndarray, table: DetectedTable) -> DetectedTable:
        """Recognize structure by detecting text regions and their alignment"""
        x1, y1, x2, y2 = table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2
        table_crop = image[y1:y2, x1:x2]
        
        if table_crop.size == 0:
            return table
        
        gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        
        # Use MSER or connected components to find text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate to merge nearby text
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        dilated = cv2.dilate(binary, h_kernel, iterations=1)
        
        # Find connected components (text lines)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)
        
        # Get Y positions of text lines (rows)
        y_positions = []
        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] > 50:
                cy = centroids[i][1]
                y_positions.append(int(cy))
        
        # Cluster Y positions into rows
        if y_positions:
            y_positions = sorted(set(y_positions))
            row_positions = self._cluster_positions(y_positions, min_gap=10)
        else:
            row_positions = [0, table_crop.shape[0]]
        
        # Similarly for columns using vertical projection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        v_dilated = cv2.dilate(binary, v_kernel, iterations=1)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(v_dilated)
        
        x_positions = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 50:
                cx = centroids[i][0]
                x_positions.append(int(cx))
        
        if x_positions:
            x_positions = sorted(set(x_positions))
            col_positions = self._cluster_positions(x_positions, min_gap=20)
        else:
            col_positions = [0, table_crop.shape[1]]
        
        # Convert to row boundaries
        row_boundaries = [0]
        for i in range(len(row_positions) - 1):
            mid = (row_positions[i] + row_positions[i + 1]) // 2
            row_boundaries.append(mid)
        row_boundaries.append(table_crop.shape[0])
        
        col_boundaries = [0]
        for i in range(len(col_positions) - 1):
            mid = (col_positions[i] + col_positions[i + 1]) // 2
            col_boundaries.append(mid)
        col_boundaries.append(table_crop.shape[1])
        
        return self._create_grid_from_positions(table, row_boundaries, col_boundaries, x1, y1)
    
    def _cluster_positions(self, positions: List[int], min_gap: int = 10) -> List[int]:
        """Cluster nearby positions together"""
        if not positions:
            return []
        
        positions = sorted(positions)
        clusters = [[positions[0]]]
        
        for pos in positions[1:]:
            if pos - clusters[-1][-1] <= min_gap:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        # Return cluster centers
        return [int(np.mean(c)) for c in clusters]
    
    def _create_grid_from_positions(self, table: DetectedTable, 
                                    h_positions: List[int], v_positions: List[int],
                                    offset_x: int, offset_y: int) -> DetectedTable:
        """Create grid structure from row/column positions"""
        rows = []
        for i in range(len(h_positions) - 1):
            row_y1 = h_positions[i]
            row_y2 = h_positions[i + 1]
            rows.append(TableRow(
                bbox=BBox(table.bbox.x1, offset_y + row_y1, table.bbox.x2, offset_y + row_y2),
                index=i
            ))
        
        columns = []
        for i in range(len(v_positions) - 1):
            col_x1 = v_positions[i]
            col_x2 = v_positions[i + 1]
            columns.append(TableColumn(
                bbox=BBox(offset_x + col_x1, table.bbox.y1, offset_x + col_x2, table.bbox.y2),
                index=i
            ))
        
        # Generate cells
        cells = []
        for row in rows:
            for col in columns:
                cell = TableCell(
                    bbox=BBox(col.bbox.x1, row.bbox.y1, col.bbox.x2, row.bbox.y2),
                    row=row.index,
                    col=col.index,
                    is_header=(row.index == 0)
                )
                cells.append(cell)
        
        table.rows = rows
        table.columns = columns
        table.cells = cells
        table.num_rows = len(rows)
        table.num_cols = len(columns)
        
        return table
    
    def _create_grid_from_cells(self, table: DetectedTable, 
                               cell_boxes: List[BBox]) -> DetectedTable:
        """Create grid from detected cell bounding boxes"""
        if not cell_boxes:
            return table
        
        # Cluster Y coordinates for rows
        y_coords = [(b.y1 + b.y2) // 2 for b in cell_boxes]
        row_centers = self._cluster_positions(y_coords, min_gap=15)
        
        # Cluster X coordinates for columns
        x_coords = [(b.x1 + b.x2) // 2 for b in cell_boxes]
        col_centers = self._cluster_positions(x_coords, min_gap=20)
        
        # Create rows
        rows = []
        for i, cy in enumerate(row_centers):
            matching = [b for b in cell_boxes if abs((b.y1 + b.y2) // 2 - cy) < 20]
            if matching:
                y1 = min(b.y1 for b in matching)
                y2 = max(b.y2 for b in matching)
                rows.append(TableRow(bbox=BBox(table.bbox.x1, y1, table.bbox.x2, y2), index=i))
        
        # Create columns
        columns = []
        for i, cx in enumerate(col_centers):
            matching = [b for b in cell_boxes if abs((b.x1 + b.x2) // 2 - cx) < 30]
            if matching:
                x1 = min(b.x1 for b in matching)
                x2 = max(b.x2 for b in matching)
                columns.append(TableColumn(bbox=BBox(x1, table.bbox.y1, x2, table.bbox.y2), index=i))
        
        # Generate cells
        cells = []
        for row in rows:
            for col in columns:
                cell = TableCell(
                    bbox=BBox(col.bbox.x1, row.bbox.y1, col.bbox.x2, row.bbox.y2),
                    row=row.index,
                    col=col.index,
                    is_header=(row.index == 0)
                )
                cells.append(cell)
        
        table.rows = rows
        table.columns = columns
        table.cells = cells
        table.num_rows = len(rows)
        table.num_cols = len(columns)
        
        return table

    def _find_line_positions(self, line_image: np.ndarray, horizontal: bool, 
                            min_gap: int = 10) -> List[int]:
        """Find positions of lines in the image"""
        if horizontal:
            projection = np.sum(line_image, axis=1)
        else:
            projection = np.sum(line_image, axis=0)
        
        # Find peaks (line positions)
        threshold = np.max(projection) * 0.3
        positions = []
        
        in_peak = False
        peak_start = 0
        
        for i, val in enumerate(projection):
            if val > threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif val <= threshold and in_peak:
                in_peak = False
                positions.append((peak_start + i) // 2)
        
        # Add edges if no lines found
        if len(positions) < 2:
            positions = [0, len(projection)]
        
        return sorted(positions)


class TableOCR:
    """Extract text from table cells using OCR"""
    
    def __init__(self, languages: List[str] = ['pt', 'en']):
        if not OCR_AVAILABLE:
            raise ImportError("easyocr required")
        
        self.reader = easyocr.Reader(languages, gpu=TORCH_AVAILABLE)
        logger.info("OCR reader initialized")
    
    def extract_cell_text(self, image: np.ndarray, table: DetectedTable) -> DetectedTable:
        """Extract text from all cells in a table"""
        for cell in table.cells:
            x1, y1, x2, y2 = cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2
            cell_crop = image[y1:y2, x1:x2]
            
            if cell_crop.size > 0:
                try:
                    results = self.reader.readtext(cell_crop)
                    if results:
                        cell.text = " ".join([r[1] for r in results])
                        cell.confidence = np.mean([r[2] for r in results])
                except Exception as e:
                    logger.debug(f"OCR error for cell: {e}")
        
        return table


class TableDetectionPipeline:
    """
    Complete pipeline for table detection and structure recognition
    """
    
    # Default trained model paths
    DEFAULT_YOLO_MODEL = "runs/detect/yolo_table_detection/train_20260122_233742/weights/best.pt"
    DEFAULT_STRUCTURE_MODEL = "structure_models/structure_model_best.pt"
    
    def __init__(self, 
                 detection_method: str = "yolo",
                 structure_method: str = "transformer",
                 use_ocr: bool = True,
                 device: str = "auto",
                 yolo_model_path: Optional[str] = None,
                 structure_model_path: Optional[str] = None):
        """
        Initialize pipeline
        
        Args:
            detection_method: "transformer", "yolo", or "both"
            structure_method: "transformer", "heuristic", "lines", "contours", 
                            "text_alignment", "auto", or "trained" (use trained model)
            use_ocr: Whether to extract cell text
            device: "auto", "cuda", or "cpu"
            yolo_model_path: Path to custom YOLO model
            structure_model_path: Path to trained structure model
        """
        self.detection_method = detection_method
        self.structure_method = structure_method
        self.use_ocr = use_ocr
        self.device = device
        
        # Initialize detectors
        self.transformer_detector = None
        self.yolo_detector = None
        self.heuristic_structure = None
        self.trained_structure = None
        self.ocr = None
        
        # Try to load YOLO first (preferred for detection on this dataset)
        if detection_method in ["yolo", "both"]:
            if YOLO_AVAILABLE:
                # Use trained model if available, otherwise fall back
                model_path = yolo_model_path
                if not model_path:
                    if Path(self.DEFAULT_YOLO_MODEL).exists():
                        model_path = self.DEFAULT_YOLO_MODEL
                    else:
                        model_path = "yolov8s.pt"
                try:
                    self.yolo_detector = YOLOTableDetector(model_path)
                except Exception as e:
                    logger.warning(f"Failed to load YOLO model: {e}")
        
        if detection_method in ["transformer", "both"] or structure_method == "transformer":
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.transformer_detector = TableTransformerDetector(device=device)
                except Exception as e:
                    logger.warning(f"Failed to load Transformer models: {e}")
        
        # Initialize trained structure model if requested
        if structure_method == "trained":
            model_path = structure_model_path or self.DEFAULT_STRUCTURE_MODEL
            if Path(model_path).exists():
                try:
                    from table_structure_trainer import TrainedStructureRecognizer
                    self.trained_structure = TrainedStructureRecognizer(model_path, device)
                    logger.info(f"Loaded trained structure model: {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load trained structure model: {e}")
                    # Fall back to heuristic
                    self.heuristic_structure = HeuristicStructureRecognizer(method="auto")
            else:
                logger.warning(f"Trained structure model not found: {model_path}")
                self.heuristic_structure = HeuristicStructureRecognizer(method="auto")
        
        # Initialize heuristic structure recognizer with appropriate method
        heuristic_methods = ["heuristic", "lines", "contours", "text_alignment", "auto"]
        if structure_method in heuristic_methods:
            # Map structure_method to heuristic method name
            if structure_method == "heuristic":
                heuristic_method = "auto"  # Use auto for general "heuristic"
            else:
                heuristic_method = structure_method
            self.heuristic_structure = HeuristicStructureRecognizer(method=heuristic_method)
        elif structure_method not in ["transformer", "trained"] and not self.transformer_detector:
            # Fallback if transformer not available
            self.heuristic_structure = HeuristicStructureRecognizer(method="auto")
        
        if use_ocr and OCR_AVAILABLE:
            try:
                self.ocr = TableOCR()
            except Exception as e:
                logger.warning(f"Failed to initialize OCR: {e}")
    
    def process(self, image_path: str, 
                detection_threshold: float = 0.7,
                structure_threshold: float = 0.5) -> List[DetectedTable]:
        """
        Process an image to detect tables and recognize structure
        
        Args:
            image_path: Path to image file
            detection_threshold: Confidence threshold for detection
            structure_threshold: Confidence threshold for structure
            
        Returns:
            List of detected tables with structure
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"Processing: {image_path}")
        
        # Detect tables
        tables = []
        
        if self.transformer_detector and self.detection_method in ["transformer", "both"]:
            tables.extend(self.transformer_detector.detect_tables(image, detection_threshold))
        
        if self.yolo_detector and self.detection_method in ["yolo", "both"]:
            yolo_tables = self.yolo_detector.detect_tables(image, detection_threshold)
            
            # Merge if using both (remove duplicates based on IoU)
            if self.detection_method == "both":
                for yt in yolo_tables:
                    is_duplicate = any(t.bbox.iou(yt.bbox) > 0.5 for t in tables)
                    if not is_duplicate:
                        tables.append(yt)
            else:
                tables = yolo_tables
        
        # Recognize structure
        for table in tables:
            if self.trained_structure and self.structure_method == "trained":
                # Use trained structure model
                result = self.trained_structure.predict(image, table.bbox.to_list())
                table.num_rows = result["num_rows"]
                table.num_cols = result["num_cols"]
                
                # Generate rows/columns/cells from positions
                self._generate_grid_from_positions(
                    table, result["row_positions"], result["col_positions"]
                )
            elif self.transformer_detector and self.structure_method == "transformer":
                self.transformer_detector.recognize_structure(
                    image, table, structure_threshold
                )
            elif self.heuristic_structure:
                self.heuristic_structure.recognize_structure(image, table)
            else:
                # Fallback to a simple heuristic if nothing else available
                fallback = HeuristicStructureRecognizer(method="auto")
                fallback.recognize_structure(image, table)
            
            # Extract text if OCR enabled
            if self.ocr:
                self.ocr.extract_cell_text(image, table)
        
        return tables
    
    def _generate_grid_from_positions(self, table: DetectedTable,
                                       row_positions: List[int],
                                       col_positions: List[int]):
        """Generate rows, columns, and cells from position lists"""
        x1, y1 = table.bbox.x1, table.bbox.y1
        
        # Ensure we have boundary positions
        if not row_positions:
            row_positions = [0, table.bbox.y2 - table.bbox.y1]
        if not col_positions:
            col_positions = [0, table.bbox.x2 - table.bbox.x1]
        
        # Generate rows
        table.rows = []
        for i in range(len(row_positions) - 1):
            row = TableRow(
                bbox=BBox(x1, y1 + row_positions[i], 
                         table.bbox.x2, y1 + row_positions[i + 1]),
                index=i,
                is_header=(i == 0)
            )
            table.rows.append(row)
        
        # Generate columns
        table.columns = []
        for i in range(len(col_positions) - 1):
            col = TableColumn(
                bbox=BBox(x1 + col_positions[i], y1,
                         x1 + col_positions[i + 1], table.bbox.y2),
                index=i
            )
            table.columns.append(col)
        
        # Generate cells
        table.cells = []
        for row in table.rows:
            for col in table.columns:
                cell = TableCell(
                    bbox=BBox(col.bbox.x1, row.bbox.y1, col.bbox.x2, row.bbox.y2),
                    row=row.index,
                    col=col.index,
                    is_header=(row.index == 0)
                )
                table.cells.append(cell)
        
        table.num_rows = len(table.rows)
        table.num_cols = len(table.columns)

    def visualize(self, image_path: str, tables: List[DetectedTable], 
                  output_path: Optional[str] = None,
                  show_cells: bool = True,
                  show_text: bool = True) -> np.ndarray:
        """
        Visualize detected tables with structure
        
        Args:
            image_path: Path to image
            tables: Detected tables
            output_path: Path to save visualization
            show_cells: Whether to show cell boundaries
            show_text: Whether to show cell text
            
        Returns:
            Visualization image
        """
        image = cv2.imread(image_path)
        vis = image.copy()
        
        # Colors
        TABLE_COLOR = (0, 255, 0)      # Green
        ROW_COLOR = (255, 200, 0)      # Cyan
        COL_COLOR = (0, 200, 255)      # Orange
        CELL_COLOR = (255, 0, 255)     # Magenta
        HEADER_COLOR = (0, 0, 255)     # Red
        TEXT_BG = (0, 0, 0)
        TEXT_FG = (255, 255, 255)
        
        for table in tables:
            # Draw table boundary
            cv2.rectangle(vis, 
                         (table.bbox.x1, table.bbox.y1),
                         (table.bbox.x2, table.bbox.y2),
                         TABLE_COLOR, 3)
            
            # Draw confidence
            conf_text = f"Table: {table.confidence:.2f}"
            cv2.putText(vis, conf_text, 
                       (table.bbox.x1, table.bbox.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, TABLE_COLOR, 2)
            
            # Draw structure info
            info_text = f"{table.num_rows}x{table.num_cols}"
            cv2.putText(vis, info_text,
                       (table.bbox.x1, table.bbox.y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, TABLE_COLOR, 2)
            
            if show_cells:
                # Draw rows (horizontal lines)
                for row in table.rows:
                    cv2.line(vis, 
                            (table.bbox.x1, row.bbox.y1),
                            (table.bbox.x2, row.bbox.y1),
                            ROW_COLOR, 1)
                
                # Draw columns (vertical lines)
                for col in table.columns:
                    cv2.line(vis,
                            (col.bbox.x1, table.bbox.y1),
                            (col.bbox.x1, table.bbox.y2),
                            COL_COLOR, 1)
                
                # Draw cells
                for cell in table.cells:
                    color = HEADER_COLOR if cell.is_header else CELL_COLOR
                    cv2.rectangle(vis,
                                 (cell.bbox.x1, cell.bbox.y1),
                                 (cell.bbox.x2, cell.bbox.y2),
                                 color, 1)
                    
                    if show_text and cell.text:
                        # Draw text with background
                        text = cell.text[:20] + "..." if len(cell.text) > 20 else cell.text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.3
                        thickness = 1
                        
                        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        tx = cell.bbox.x1 + 2
                        ty = cell.bbox.y1 + th + 2
                        
                        cv2.rectangle(vis, (tx - 1, ty - th - 1),
                                     (tx + tw + 1, ty + 1), TEXT_BG, -1)
                        cv2.putText(vis, text, (tx, ty), font, font_scale, TEXT_FG, thickness)
        
        if output_path:
            cv2.imwrite(output_path, vis)
            logger.info(f"Visualization saved: {output_path}")
        
        return vis


def load_ground_truth(json_path: str) -> List[Dict]:
    """Load ground truth annotations from LabelMe JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tables = []
    for shape in data.get('shapes', []):
        if shape.get('label', '').lower().startswith('table'):
            points = shape['points']
            x1 = int(min(p[0] for p in points))
            y1 = int(min(p[1] for p in points))
            x2 = int(max(p[0] for p in points))
            y2 = int(max(p[1] for p in points))
            
            tables.append({
                'label': shape['label'],
                'bbox': [x1, y1, x2, y2]
            })
    
    return tables


def evaluate_detection(predictions: List[DetectedTable], 
                      ground_truth: List[Dict],
                      iou_threshold: float = 0.5) -> Dict:
    """Evaluate detection results against ground truth"""
    pred_boxes = [p.bbox for p in predictions]
    gt_boxes = [BBox(*gt['bbox']) for gt in ground_truth]
    
    # Find matches
    matches = []
    used_gt = set()
    used_pred = set()
    
    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt = -1
        
        for j, gt in enumerate(gt_boxes):
            if j in used_gt:
                continue
            
            iou = pred.iou(gt)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt = j
        
        if best_gt >= 0:
            matches.append((i, best_gt, best_iou))
            used_gt.add(best_gt)
            used_pred.add(i)
    
    # Calculate metrics
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_iou": np.mean([m[2] for m in matches]) if matches else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Table Detection and Structure Recognition")
    parser.add_argument("--input", "-i", help="Input image path")
    parser.add_argument("--output", "-o", help="Output visualization path")
    parser.add_argument("--json-output", help="Output JSON path for structure")
    parser.add_argument("--detection", default="yolo", 
                       choices=["transformer", "yolo", "both"],
                       help="Detection method (default: yolo with trained model)")
    parser.add_argument("--structure", default="auto",
                       choices=["transformer", "heuristic", "lines", "contours", "text_alignment", "auto", "trained"],
                       help="Structure recognition method: transformer (ML), lines (detect grid lines), "
                            "contours (find cell boundaries), text_alignment (cluster text regions), "
                            "auto (try all heuristics), trained (use custom trained model)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--yolo-model", help="Path to custom YOLO model (uses trained model by default)")
    parser.add_argument("--structure-model", help="Path to trained structure model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--batch", help="Process all images in directory")
    parser.add_argument("--evaluate", help="Ground truth JSON for evaluation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TableDetectionPipeline(
        detection_method=args.detection,
        structure_method=args.structure,
        use_ocr=not args.no_ocr,
        device=args.device,
        yolo_model_path=args.yolo_model,
        structure_model_path=getattr(args, 'structure_model', None)
    )
    
    if args.batch:
        # Batch processing
        input_dir = Path(args.batch)
        output_dir = Path(args.output) if args.output else input_dir / "table_detections"
        output_dir.mkdir(exist_ok=True)
        
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        
        all_results = {}
        for img_file in image_files:
            try:
                tables = pipeline.process(str(img_file), args.threshold)
                
                # Save visualization
                vis_path = output_dir / f"{img_file.stem}_tables.png"
                pipeline.visualize(str(img_file), tables, str(vis_path))
                
                # Save structure JSON
                json_path = output_dir / f"{img_file.stem}_structure.json"
                with open(json_path, 'w') as f:
                    json.dump({
                        "source": str(img_file),
                        "tables": [t.to_dict() for t in tables]
                    }, f, indent=2)
                
                all_results[str(img_file)] = [t.to_dict() for t in tables]
                
                # Evaluate if ground truth provided
                if args.evaluate:
                    gt_path = input_dir / f"{img_file.stem}.json"
                    if gt_path.exists():
                        gt = load_ground_truth(str(gt_path))
                        metrics = evaluate_detection(tables, gt)
                        logger.info(f"  Metrics: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
        
        logger.info(f"✓ Processed {len(image_files)} images")
        
    elif args.input:
        # Single image
        tables = pipeline.process(args.input, args.threshold)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Found {len(tables)} tables")
        print(f"{'='*60}")
        
        for i, table in enumerate(tables):
            print(f"\nTable {i+1}:")
            print(f"  Confidence: {table.confidence:.2f}")
            print(f"  Bounding box: {table.bbox.to_list()}")
            print(f"  Structure: {table.num_rows} rows x {table.num_cols} columns")
            
            if table.cells:
                print(f"  Cells: {len(table.cells)}")
                
                # Print as grid
                grid = table.to_grid()
                if grid:
                    print("\n  Content:")
                    for row in grid[:5]:  # First 5 rows
                        print(f"    | {' | '.join(cell[:15] for cell in row)} |")
                    if len(grid) > 5:
                        print(f"    ... ({len(grid) - 5} more rows)")
        
        # Save visualization
        if args.output:
            pipeline.visualize(args.input, tables, args.output)
        
        # Save JSON
        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump({
                    "source": args.input,
                    "tables": [t.to_dict() for t in tables]
                }, f, indent=2)
        
        # Evaluate
        if args.evaluate:
            gt = load_ground_truth(args.evaluate)
            metrics = evaluate_detection(tables, gt)
            print(f"\nEvaluation:")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1 Score: {metrics['f1']:.2f}")
            print(f"  Mean IoU: {metrics['mean_iou']:.2f}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
