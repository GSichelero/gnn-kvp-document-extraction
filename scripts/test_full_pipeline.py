#!/usr/bin/env python3
"""
Full Pipeline Integration Test
Tests all 5 stages of the unified document pipeline on real invoice images
and generates comprehensive visualizations.

Usage:
    python test_full_pipeline.py                              # Test on default image
    python test_full_pipeline.py --image path/to/image.jpg    # Test on specific image
    python test_full_pipeline.py --batch 5                    # Test on first 5 images
    python test_full_pipeline.py --all                        # Test all images
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Color palette
COLORS = {
    'table': (220, 50, 50),       # Red
    'row': (50, 120, 220),        # Blue
    'column': (50, 180, 80),      # Green
    'header': (180, 50, 180),     # Purple
    'cell': (220, 160, 30),       # Orange
    'key': (255, 140, 0),         # Dark Orange
    'value': (138, 43, 226),      # Blue Violet
    'link': (0, 200, 200),        # Cyan
    'ocr': (100, 100, 100),       # Gray
    'category': (34, 139, 34),    # Forest Green
}


def draw_rounded_rect(draw, bbox, color, width=2, radius=4):
    """Draw a rounded rectangle"""
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def create_legend(width, stats: Dict) -> Image.Image:
    """Create a legend panel with statistics"""
    height = 40 + len(stats) * 22
    legend = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(legend)
    
    y = 10
    draw.text((10, y), "Pipeline Results Summary", fill='black')
    y += 25
    
    for label, (value, color) in stats.items():
        # Color swatch
        draw.rectangle([10, y + 2, 25, y + 14], fill=color, outline='black')
        draw.text((32, y), f"{label}: {value}", fill='black')
        y += 22
    
    return legend


def visualize_single_panel(image: Image.Image, result_data: Dict, output_path: str):
    """Create a single-panel visualization with all results overlaid"""
    vis = image.copy()
    draw = ImageDraw.Draw(vis, 'RGBA')
    
    # Draw OCR blocks (semi-transparent gray)
    for ocr in result_data.get('ocr_results', []):
        bbox = ocr['bbox']
        overlay = Image.new('RGBA', vis.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bbox, fill=(200, 200, 200, 40), outline=(150, 150, 150, 100))
        vis = Image.alpha_composite(vis.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(vis)
    
    # Draw tables with structure
    for i, table in enumerate(result_data.get('tables', [])):
        bbox = table['bbox']
        # Table border
        draw.rectangle(bbox, outline=COLORS['table'], width=3)
        draw.text((bbox[0] + 5, bbox[1] - 15), f"Table {i+1}", fill=COLORS['table'])
        
        # Rows
        for row in table.get('rows', []):
            draw.rectangle(row, outline=COLORS['row'], width=1)
        
        # Columns
        for col in table.get('columns', []):
            draw.rectangle(col, outline=COLORS['column'], width=1)
        
        # Headers
        for hdr in table.get('headers', []):
            draw.rectangle(hdr, outline=COLORS['header'], width=2)
    
    # Draw KVP links
    for kvp in result_data.get('key_value_pairs', []):
        key_bbox = kvp.get('key_bbox')
        val_bbox = kvp.get('value_bbox')
        
        if key_bbox:
            draw.rectangle(key_bbox, outline=COLORS['key'], width=2)
        if val_bbox:
            draw.rectangle(val_bbox, outline=COLORS['value'], width=2)
        
        # Draw linking arrow
        if key_bbox and val_bbox:
            kx = (key_bbox[0] + key_bbox[2]) // 2
            ky = (key_bbox[1] + key_bbox[3]) // 2
            vx = (val_bbox[0] + val_bbox[2]) // 2
            vy = (val_bbox[1] + val_bbox[3]) // 2
            draw.line([(kx, ky), (vx, vy)], fill=COLORS['link'], width=1)
    
    vis.save(output_path)
    return vis


def visualize_multi_panel(image: Image.Image, result_data: Dict, output_path: str):
    """Create a 2x2 multi-panel visualization"""
    w, h = image.size
    panel_w, panel_h = w, h
    
    canvas = Image.new('RGB', (panel_w * 2 + 20, panel_h * 2 + 60), 'white')
    
    titles = ['(a) OCR Detections', '(b) Table Structure', 
              '(c) Key-Value Pairs', '(d) Field Categories']
    
    panels = []
    for title_idx in range(4):
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        panels.append((panel, draw))
    
    # Panel (a): OCR
    panel_a, draw_a = panels[0]
    for ocr in result_data.get('ocr_results', []):
        bbox = ocr['bbox']
        conf = ocr.get('confidence', 0)
        color = (0, int(200 * conf), 0) if conf > 0.5 else (200, 0, 0)
        draw_a.rectangle(bbox, outline=color, width=1)
    
    # Panel (b): Tables
    panel_b, draw_b = panels[1]
    for i, table in enumerate(result_data.get('tables', [])):
        draw_b.rectangle(table['bbox'], outline=COLORS['table'], width=3)
        for row in table.get('rows', []):
            draw_b.rectangle(row, outline=COLORS['row'], width=1)
        for col in table.get('columns', []):
            draw_b.rectangle(col, outline=COLORS['column'], width=1)
        for hdr in table.get('headers', []):
            draw_b.rectangle(hdr, outline=COLORS['header'], width=2)
    
    # Panel (c): KVPs
    panel_c, draw_c = panels[2]
    for kvp in result_data.get('key_value_pairs', []):
        key_bbox = kvp.get('key_bbox')
        val_bbox = kvp.get('value_bbox')
        if key_bbox:
            draw_c.rectangle(key_bbox, outline=COLORS['key'], width=2)
        if val_bbox:
            draw_c.rectangle(val_bbox, outline=COLORS['value'], width=2)
        if key_bbox and val_bbox:
            kx = (key_bbox[0] + key_bbox[2]) // 2
            ky = (key_bbox[1] + key_bbox[3]) // 2
            vx = (val_bbox[0] + val_bbox[2]) // 2
            vy = (val_bbox[1] + val_bbox[3]) // 2
            draw_c.line([(kx, ky), (vx, vy)], fill=COLORS['link'], width=1)
    
    # Panel (d): Categories
    panel_d, draw_d = panels[3]
    cat_colors = {}
    color_list = [(34, 139, 34), (220, 20, 60), (30, 144, 255), (255, 165, 0),
                  (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127)]
    
    for kvp in result_data.get('key_value_pairs', []):
        cat = kvp.get('category', 'unknown')
        if cat not in cat_colors:
            cat_colors[cat] = color_list[len(cat_colors) % len(color_list)]
        color = cat_colors[cat]
        
        val_bbox = kvp.get('value_bbox')
        if val_bbox:
            draw_d.rectangle(val_bbox, outline=color, width=2)
            draw_d.text((val_bbox[0], val_bbox[1] - 12), cat[:15], fill=color)
    
    # Compose
    for idx, (panel, _) in enumerate(panels):
        col = idx % 2
        row_idx = idx // 2
        x = col * (panel_w + 10) + 5
        y = row_idx * (panel_h + 30) + 25
        
        # Title
        title_draw = ImageDraw.Draw(canvas)
        title_draw.text((x + 10, y - 20), titles[idx], fill='black')
        
        canvas.paste(panel, (x, y))
    
    canvas.save(output_path)
    return canvas


def print_results_summary(result_data: Dict, image_path: str, processing_time: float):
    """Print a formatted summary of pipeline results"""
    print(f"\n{'='*70}")
    print(f"📋 PIPELINE RESULTS: {Path(image_path).name}")
    print(f"{'='*70}")
    
    print(f"\n⏱️  Processing time: {processing_time:.2f}s")
    
    # OCR
    ocr = result_data.get('ocr_results', [])
    print(f"\n📝 OCR: {len(ocr)} text blocks detected")
    if ocr:
        avg_conf = sum(o.get('confidence', 0) for o in ocr) / len(ocr)
        print(f"   Average confidence: {avg_conf:.3f}")
    
    # Tables
    tables = result_data.get('tables', [])
    print(f"\n📊 Tables: {len(tables)} detected")
    for i, t in enumerate(tables):
        rows = len(t.get('rows', []))
        cols = len(t.get('columns', []))
        hdrs = len(t.get('headers', []))
        print(f"   Table {i+1}: {rows} rows × {cols} columns, {hdrs} headers")
    
    # KVPs
    kvps = result_data.get('key_value_pairs', [])
    print(f"\n🔗 Key-Value Pairs: {len(kvps)} extracted")
    for kvp in kvps[:10]:
        cat = kvp.get('category', '?')
        conf = kvp.get('confidence', 0)
        print(f"   [{cat}] {kvp['key'][:30]}: {kvp['value'][:30]} (conf={conf:.2f})")
    if len(kvps) > 10:
        print(f"   ... and {len(kvps) - 10} more")
    
    # Categories
    categories = result_data.get('classified_fields', {})
    print(f"\n🏷️  Categories: {len(categories)} identified")
    for cat, fields in sorted(categories.items()):
        field_count = len(fields) if isinstance(fields, list) else 1
        print(f"   {cat}: {field_count} field(s)")
    
    print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_pipeline_on_image(image_path: str, output_dir: str, pipeline=None) -> Dict:
    """
    Run the full pipeline on a single image and save results + visualizations.
    
    Returns:
        Dictionary with result data
    """
    from unified_document_pipeline import UnifiedDocumentPipeline
    
    image_path = str(Path(image_path).resolve())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(image_path).stem
    
    # Initialize pipeline if not provided
    if pipeline is None:
        logger.info("Initializing pipeline...")
        pipeline = UnifiedDocumentPipeline()
    
    # Process image
    logger.info(f"Processing: {image_path}")
    start = time.time()
    result = pipeline.process(image_path, save_visualization=False)
    processing_time = time.time() - start
    
    # Save results JSON
    json_path = output_dir / f"{stem}_results.json"
    pipeline.save_results(result, str(json_path))
    
    # Load the result data for visualization
    with open(json_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    result_data['processing_time'] = processing_time
    
    # Print summary
    print_results_summary(result_data, image_path, processing_time)
    
    # Generate visualizations
    image = Image.open(image_path).convert('RGB')
    
    # Single panel
    single_path = output_dir / f"{stem}_visualization.png"
    visualize_single_panel(image, result_data, str(single_path))
    logger.info(f"📸 Single panel: {single_path}")
    
    # Multi panel
    multi_path = output_dir / f"{stem}_multi_panel.png"
    visualize_multi_panel(image, result_data, str(multi_path))
    logger.info(f"📸 Multi panel: {multi_path}")
    
    return result_data


def test_batch(image_dir: str, output_dir: str, max_images: int = 5) -> List[Dict]:
    """Test pipeline on multiple images"""
    from unified_document_pipeline import UnifiedDocumentPipeline
    
    image_dir = Path(image_dir)
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    if max_images > 0:
        images = images[:max_images]
    
    logger.info(f"Testing on {len(images)} images from {image_dir}")
    
    # Initialize pipeline once
    pipeline = UnifiedDocumentPipeline()
    
    results = []
    success_count = 0
    total_time = 0
    total_kvps = 0
    total_tables = 0
    total_categories = set()
    
    for i, img_path in enumerate(images):
        print(f"\n{'━'*60}")
        print(f"  Image {i+1}/{len(images)}: {img_path.name}")
        print(f"{'━'*60}")
        
        try:
            result_data = test_pipeline_on_image(
                str(img_path), output_dir, pipeline=pipeline
            )
            results.append(result_data)
            success_count += 1
            total_time += result_data.get('processing_time', 0)
            total_kvps += len(result_data.get('key_value_pairs', []))
            total_tables += len(result_data.get('tables', []))
            for cat in result_data.get('classified_fields', {}):
                total_categories.add(cat)
                
        except Exception as e:
            logger.error(f"❌ Failed on {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print batch summary
    print(f"\n{'═'*70}")
    print(f"📊 BATCH TEST SUMMARY")
    print(f"{'═'*70}")
    print(f"  Images processed: {success_count}/{len(images)}")
    print(f"  Success rate: {success_count/len(images)*100:.0f}%")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/image: {total_time/max(success_count,1):.1f}s")
    print(f"  Total KVPs: {total_kvps} (avg {total_kvps/max(success_count,1):.0f}/image)")
    print(f"  Total tables: {total_tables} (avg {total_tables/max(success_count,1):.1f}/image)")
    print(f"  Unique categories: {len(total_categories)}")
    print(f"{'═'*70}\n")
    
    # Save batch summary
    summary = {
        'images_tested': len(images),
        'success_count': success_count,
        'total_time': total_time,
        'avg_time': total_time / max(success_count, 1),
        'total_kvps': total_kvps,
        'total_tables': total_tables,
        'categories': sorted(list(total_categories)),
    }
    summary_path = Path(output_dir) / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline Integration Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_full_pipeline.py                           # Default image
    python test_full_pipeline.py --image "faturas-images/01 2020.pdf_0.jpg"
    python test_full_pipeline.py --batch 5                 # First 5 images
    python test_full_pipeline.py --batch 0 --all           # All images
"""
    )
    
    parser.add_argument("--image", type=str, help="Path to specific image")
    parser.add_argument("--batch", type=int, default=0, help="Number of images for batch test (0=disabled)")
    parser.add_argument("--all", action="store_true", help="Test all images")
    parser.add_argument("--output", type=str, default="pipeline_results", help="Output directory")
    parser.add_argument("--images_dir", type=str, default="faturas-images", help="Images directory for batch")
    
    args = parser.parse_args()
    
    # Determine what to test
    if args.all:
        test_batch(args.images_dir, args.output, max_images=-1)
    elif args.batch > 0:
        test_batch(args.images_dir, args.output, max_images=args.batch)
    elif args.image:
        test_pipeline_on_image(args.image, args.output)
    else:
        # Default: test on first available image
        images_dir = Path(args.images_dir)
        images = sorted(list(images_dir.glob("*.jpg")))
        if images:
            test_pipeline_on_image(str(images[0]), args.output)
        else:
            print("❌ No images found. Use --image to specify one.")
            sys.exit(1)


if __name__ == "__main__":
    main()
