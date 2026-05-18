#!/usr/bin/env python3
"""Auxiliary locator classifier using Sentence Transformers.

This optional tool classifies locators from ``default_locators.json`` into
semantic categories inferred from local ground-truth category assignments.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from invoice_categories import build_category_vocabulary


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install: "
        "pip install sentence-transformers scikit-learn matplotlib pandas"
    ) from exc


class LocatorClassifier:
    """Classifier for optional locator-analysis scripts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info("Loading sentence transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)

        self.predefined_categories = build_category_vocabulary()
        if not self.predefined_categories:
            logger.warning(
                "No GT category vocabulary found; locator classification will "
                "fall back to the unclassified bucket."
            )

        self.category_embeddings = self._generate_category_embeddings()

    def _generate_category_embeddings(self) -> Dict[str, np.ndarray]:
        category_embeddings = {}
        for category, keywords in self.predefined_categories.items():
            description = " ".join(keywords)
            category_embeddings[category] = self.model.encode([description])[0]
        return category_embeddings

    def load_locators(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_all_locators(self, locators_data: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        all_locators = []
        for file_name, file_data in locators_data.items():
            if not isinstance(file_data, dict):
                continue

            if "locators" in file_data:
                for locator in file_data["locators"]:
                    all_locators.append((locator, file_name, 1.0))
            elif "line_locators" in file_data:
                for locator, weight in file_data["line_locators"].items():
                    all_locators.append((f"{locator} (line)", file_name, weight))
                for locator, weight in file_data["column_locators"].items():
                    all_locators.append((f"{locator} (column)", file_name, weight))
            else:
                for locator, weight in file_data.items():
                    if locator not in {"note", "additional_note"}:
                        all_locators.append((locator, file_name, weight))
        return all_locators

    def classify_locators(self, locators: List[Tuple[str, str, float]]) -> pd.DataFrame:
        logger.info("Classifying %d locators", len(locators))
        if not locators:
            return pd.DataFrame(
                columns=[
                    "locator",
                    "source_file",
                    "weight",
                    "predicted_category",
                    "confidence",
                    "all_similarities",
                ]
            )

        locator_texts = [loc[0] for loc in locators]
        locator_embeddings = self.model.encode(locator_texts, show_progress_bar=True)

        results = []
        for i, (locator, source_file, weight) in enumerate(locators):
            similarities = {}
            for category, category_embedding in self.category_embeddings.items():
                similarity = cosine_similarity([locator_embeddings[i]], [category_embedding])[0][0]
                similarities[category] = similarity

            if similarities:
                best_category = max(similarities, key=similarities.get)
                best_score = similarities[best_category]
            else:
                best_category = "unclassified"
                best_score = 0.0

            results.append(
                {
                    "locator": locator,
                    "source_file": source_file,
                    "weight": weight,
                    "predicted_category": best_category,
                    "confidence": best_score,
                    "all_similarities": similarities,
                }
            )

        return pd.DataFrame(results)

    def cluster_locators(self, locators: List[Tuple[str, str, float]], n_clusters: int = 8) -> pd.DataFrame:
        logger.info("Clustering %d locators", len(locators))
        if not locators:
            return pd.DataFrame(columns=["locator", "source_file", "weight", "cluster", "embedding"])

        locator_texts = [loc[0] for loc in locators]
        embeddings = self.model.encode(locator_texts, show_progress_bar=True)
        n_clusters = max(1, min(n_clusters, len(locators)))
        cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embeddings)

        results = []
        for i, (locator, source_file, weight) in enumerate(locators):
            results.append(
                {
                    "locator": locator,
                    "source_file": source_file,
                    "weight": weight,
                    "cluster": cluster_labels[i],
                    "embedding": embeddings[i],
                }
            )

        df = pd.DataFrame(results)
        df.attrs["cluster_summaries"] = {
            cluster_id: {
                "size": len(df[df["cluster"] == cluster_id]),
                "examples": df[df["cluster"] == cluster_id]["locator"].head(5).tolist(),
                "common_words": self._extract_common_words(
                    df[df["cluster"] == cluster_id]["locator"].tolist()
                ),
            }
            for cluster_id in sorted(df["cluster"].unique())
        }
        return df

    def _extract_common_words(self, locators: List[str]) -> List[str]:
        import re

        words = []
        for locator in locators:
            words.extend(re.findall(r"\b\w+\b", locator.lower()))
        return [word for word, _ in Counter(words).most_common(5) if len(word) > 2]

    def visualize_clusters(self, df: pd.DataFrame, output_path: str = None) -> None:
        if df.empty:
            logger.warning("No clusters to visualize")
            return

        embeddings = np.vstack(df["embedding"].values)
        embeddings_2d = PCA(n_components=2).fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=df["cluster"], cmap="tab10", alpha=0.6)
        plt.colorbar(scatter)
        plt.title("Locator Clusters Visualization (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Visualization saved to: %s", output_path)
        else:
            plt.show()

    def generate_report(self, classification_df: pd.DataFrame, clustering_df: pd.DataFrame = None) -> str:
        report = ["LOCATOR CLASSIFICATION REPORT", "=" * 50]
        report.append(f"Total locators analyzed: {len(classification_df)}")

        if not classification_df.empty:
            category_counts = classification_df["predicted_category"].value_counts()
            report.append(f"Categories found: {classification_df['predicted_category'].nunique()}")
            report.append("\nCATEGORY DISTRIBUTION:")
            for category, count in category_counts.items():
                percentage = (count / len(classification_df)) * 100
                report.append(f"  {category}: {count} ({percentage:.1f}%)")

            report.append("\nEXAMPLES BY CATEGORY:")
            for category in category_counts.index[:5]:
                examples = classification_df[classification_df["predicted_category"] == category]["locator"].head(3)
                report.append(f"  {category}:")
                for example in examples:
                    report.append(f"    - {example}")

        if clustering_df is not None and not clustering_df.empty:
            report.append("\nCLUSTERING SUMMARY:")
            for cluster_id, summary in clustering_df.attrs.get("cluster_summaries", {}).items():
                report.append(f"  Cluster {cluster_id}: {summary['size']} items")
                report.append(f"    Examples: {', '.join(summary['examples'])}")
                report.append(f"    Common words: {', '.join(summary['common_words'])}")

        return "\n".join(report)

    def save_results(
        self,
        classification_df: pd.DataFrame,
        clustering_df: pd.DataFrame = None,
        output_dir: str = "locator_analysis_output",
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        classification_df.to_csv(output_path / "classification_results.csv", index=False)
        if clustering_df is not None:
            clustering_df.drop(columns=["embedding"], errors="ignore").to_csv(
                output_path / "clustering_results.csv", index=False
            )
            clustering_df.to_pickle(output_path / "clustering_results.pkl")

        report = self.generate_report(classification_df, clustering_df)
        (output_path / "analysis_report.txt").write_text(report, encoding="utf-8")
        logger.info("Results saved to: %s", output_path)


def main():
    logger.info("Starting locator classification analysis")
    classifier = LocatorClassifier()

    locators_file = Path("default_locators.json")
    if not locators_file.exists():
        logger.error("Locators file not found: %s", locators_file)
        return

    locators_data = classifier.load_locators(str(locators_file))
    all_locators = classifier.extract_all_locators(locators_data)
    classification_results = classifier.classify_locators(all_locators)
    clustering_results = classifier.cluster_locators(all_locators, n_clusters=10)
    classifier.visualize_clusters(clustering_results, "locator_analysis_output/clusters_visualization.png")
    classifier.save_results(classification_results, clustering_results)


if __name__ == "__main__":
    main()
