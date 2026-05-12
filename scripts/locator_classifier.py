#!/usr/bin/env python3
"""
Brazilian Energy Bill Locator Classifier using Sentence Transformers

This tool classifies locators from default_locators.json into semantic categories
specific to Brazilian energy bills (faturas de energia elétrica).

Categories include all the key fields typically found in Brazilian energy bills:
- Customer identification (Cod Cliente, CNPJ, Ligação)
- Account information (Conta Contrato, Instalação, Medidor)
- Billing periods and readings (Leitura Anterior/Atual/Próxima, Mês Ref)
- Financial details (Total Fatura, Vencimento, Nota Fiscal, Série NF)
- Address information (Logradouro, Cidade, Estado, CEP)
- Technical specifications (Demanda Contratada Ponta/Fora Ponta)
- Banking information (Código Débito Automático, Banco, Agência)
- Administrative data (Data Emissão, Distribuidora, Código de Barras)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ All dependencies loaded successfully")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Please install: pip install sentence-transformers scikit-learn matplotlib seaborn pandas")
    exit(1)

class LocatorClassifier:
    """
    Classifier for Brazilian energy bill locators using sentence transformers.
    
    This classifier identifies and categorizes specific fields typically found in
    Brazilian energy bills (faturas de energia elétrica) including customer information,
    billing details, meter readings, and administrative data.
    
    Categories include:
    - Customer identification (Cod Cliente, CNPJ, etc.)
    - Account information (Conta Contrato, Instalação, etc.)
    - Meter readings (Leitura Anterior, Atual, Próxima)
    - Billing details (Total Fatura, Vencimento, etc.)
    - Address information (Logradouro, Cidade, Estado, CEP)
    - Technical data (Medidor, Ligação, Demanda Contratada)
    - Administrative info (Nota Fiscal, Série NF, Distribuidora)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the classifier
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        print(f"🔄 Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Model loaded successfully")
        
        # Predefined categories for Brazilian energy bill locators
        self.predefined_categories = {
            "cod_cliente": [
                "código cliente", "cod cliente", "código do cliente", "identificação cliente",
                "customer code", "id cliente", "código consumidor"
            ],
            "conta_contrato": [
                "conta contrato", "número contrato", "conta", "contrato",
                "account contract", "contract number", "número da conta"
            ],
            "instalacao": [
                "instalação", "unidade consumidora", "instalacao", "uc",
                "installation", "consumer unit", "código instalação"
            ],
            "leitura_anterior": [
                "leitura anterior", "medição anterior", "leitura ant",
                "previous reading", "última leitura", "anterior"
            ],
            "leitura_atual": [
                "leitura atual", "medição atual", "leitura corrente",
                "current reading", "leitura presente", "atual"
            ],
            "proxima_leitura": [
                "próxima leitura", "proxima leitura", "próximo período",
                "next reading", "leitura seguinte", "próxima"
            ],
            "mes_ref": [
                "mês referência", "mes ref", "período referência", "referência",
                "reference month", "billing period", "período faturamento"
            ],
            "vencimento": [
                "vencimento", "data vencimento", "vence em", "due date",
                "prazo", "validade", "data limite"
            ],
            "total_fatura": [
                "total fatura", "valor total", "total r$", "valor pagar",
                "bill total", "amount due", "total a pagar", "valor devido"
            ],
            "nota_fiscal": [
                "número nota fiscal", "nº nota fiscal", "nota fiscal",
                "invoice number", "nf", "número nf"
            ],
            "serie_nf": [
                "série nota fiscal", "série nf", "serie nf",
                "invoice series", "série", "series"
            ],
            "data_emissao": [
                "data emissão", "data de emissão", "emissão",
                "issue date", "emission date", "data fatura"
            ],
            "dem_contratada_ponta": [
                "demanda contratada ponta", "dem contratada ponta", "demanda ponta",
                "contracted demand peak", "peak demand", "ponta"
            ],
            "dem_contratada_fora_ponta": [
                "demanda contratada fora ponta", "dem contratada fora ponta",
                "off peak demand", "fora ponta", "demanda fora ponta"
            ],
            "numero": [
                "número", "nº", "num", "number", "n°", "nr"
            ],
            "logradouro": [
                "logradouro", "rua", "avenida", "alameda", "travessa",
                "street", "address", "endereço", "via"
            ],
            "cidade": [
                "cidade", "município", "city", "localidade", "municipio"
            ],
            "estado": [
                "estado", "uf", "state", "unidade federativa"
            ],
            "cep": [
                "cep", "código postal", "postal code", "zip code"
            ],
            "codigo_barras": [
                "código de barras", "codigo de barras", "código barras",
                "barcode", "código pagamento", "linha digitável"
            ],
            "cod_debito_automatico": [
                "código débito automático", "cod débito automático",
                "automatic debit code", "débito automático", "código autorização"
            ],
            "banco": [
                "banco", "bank", "instituição financeira", "instituição bancária"
            ],
            "agencia": [
                "agência", "agencia", "agency", "branch", "código agência"
            ],
            "filename": [
                "filename", "nome arquivo", "arquivo", "file", "documento"
            ],
            "distribuidora": [
                "distribuidora", "empresa", "companhia", "concessionária",
                "distributor", "utility company", "fornecedora"
            ],
            "medidor": [
                "número medidor", "nº medidor", "medidor", "meter",
                "código medidor", "identificação medidor"
            ],
            "cnpj": [
                "cnpj", "cadastro nacional pessoa jurídica", "registro empresa",
                "company registration", "tax id"
            ],
            "ligacao": [
                "ligação", "conexão", "tipo ligação", "ligacao",
                "connection", "service connection", "ramal"
            ]
        }
        
        # Generate embeddings for category descriptions
        self.category_embeddings = self._generate_category_embeddings()
        
    def _generate_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for predefined categories"""
        category_embeddings = {}
        
        for category, keywords in self.predefined_categories.items():
            # Combine keywords into a description
            description = " ".join(keywords)
            embedding = self.model.encode([description])[0]
            category_embeddings[category] = embedding
            
        return category_embeddings
    
    def load_locators(self, file_path: str) -> Dict[str, Any]:
        """
        Load locators from JSON file
        
        Args:
            file_path: Path to the locators JSON file
            
        Returns:
            Loaded locators data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_all_locators(self, locators_data: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """
        Extract all locators with their source file and weight
        
        Args:
            locators_data: Loaded locators data
            
        Returns:
            List of (locator, source_file, weight) tuples
        """
        all_locators = []
        
        for file_name, file_data in locators_data.items():
            if isinstance(file_data, dict):
                # Handle different file formats
                if "locators" in file_data:
                    # List format (implicit weight = 1)
                    for locator in file_data["locators"]:
                        all_locators.append((locator, file_name, 1.0))
                elif "line_locators" in file_data:
                    # 2D format with line and column locators
                    for locator, weight in file_data["line_locators"].items():
                        all_locators.append((f"{locator} (line)", file_name, weight))
                    for locator, weight in file_data["column_locators"].items():
                        all_locators.append((f"{locator} (column)", file_name, weight))
                else:
                    # Standard format with weights
                    for locator, weight in file_data.items():
                        if locator not in ["note", "additional_note"]:
                            all_locators.append((locator, file_name, weight))
        
        return all_locators
    
    def classify_locators(self, locators: List[Tuple[str, str, float]]) -> pd.DataFrame:
        """
        Classify locators into categories using sentence transformers
        
        Args:
            locators: List of (locator, source_file, weight) tuples
            
        Returns:
            DataFrame with classification results
        """
        logger.info(f"🔍 Classifying {len(locators)} locators...")
        
        # Extract just the locator text for embedding
        locator_texts = [loc[0] for loc in locators]
        
        # Generate embeddings for all locators
        print("🔄 Generating embeddings...")
        locator_embeddings = self.model.encode(locator_texts, show_progress_bar=True)
        
        # Classify each locator
        results = []
        
        for i, (locator, source_file, weight) in enumerate(locators):
            locator_embedding = locator_embeddings[i]
            
            # Calculate similarity with each category
            similarities = {}
            for category, category_embedding in self.category_embeddings.items():
                similarity = cosine_similarity([locator_embedding], [category_embedding])[0][0]
                similarities[category] = similarity
            
            # Find best matching category
            best_category = max(similarities, key=similarities.get)
            best_score = similarities[best_category]
            
            results.append({
                'locator': locator,
                'source_file': source_file,
                'weight': weight,
                'predicted_category': best_category,
                'confidence': best_score,
                'all_similarities': similarities
            })
        
        return pd.DataFrame(results)
    
    def cluster_locators(self, locators: List[Tuple[str, str, float]], n_clusters: int = 8) -> pd.DataFrame:
        """
        Cluster locators using K-means on embeddings
        
        Args:
            locators: List of (locator, source_file, weight) tuples
            n_clusters: Number of clusters to create
            
        Returns:
            DataFrame with clustering results
        """
        logger.info(f"🎯 Clustering {len(locators)} locators into {n_clusters} clusters...")
        
        # Extract locator texts and generate embeddings
        locator_texts = [loc[0] for loc in locators]
        embeddings = self.model.encode(locator_texts, show_progress_bar=True)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Create results DataFrame
        results = []
        for i, (locator, source_file, weight) in enumerate(locators):
            results.append({
                'locator': locator,
                'source_file': source_file,
                'weight': weight,
                'cluster': cluster_labels[i],
                'embedding': embeddings[i]
            })
        
        df = pd.DataFrame(results)
        
        # Add cluster summaries
        cluster_summaries = {}
        for cluster_id in range(n_clusters):
            cluster_locators = df[df['cluster'] == cluster_id]['locator'].tolist()
            cluster_summaries[cluster_id] = {
                'size': len(cluster_locators),
                'examples': cluster_locators[:5],  # First 5 examples
                'common_words': self._extract_common_words(cluster_locators)
            }
        
        df.attrs['cluster_summaries'] = cluster_summaries
        return df
    
    def _extract_common_words(self, locators: List[str]) -> List[str]:
        """Extract common words from a list of locators"""
        from collections import Counter
        import re
        
        # Extract words from all locators
        all_words = []
        for locator in locators:
            # Simple word extraction (lowercase, remove special chars)
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', locator.lower())
            all_words.extend(words)
        
        # Return most common words
        common = Counter(all_words).most_common(5)
        return [word for word, count in common if len(word) > 2]
    
    def visualize_clusters(self, df: pd.DataFrame, output_path: str = None) -> None:
        """
        Visualize clusters using PCA
        
        Args:
            df: DataFrame with clustering results
            output_path: Path to save the plot
        """
        # Extract embeddings
        embeddings = np.vstack(df['embedding'].values)
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=df['cluster'], cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Locator Clusters Visualization (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add cluster labels
        for cluster_id in df['cluster'].unique():
            cluster_points = embeddings_2d[df['cluster'] == cluster_id]
            centroid = cluster_points.mean(axis=0)
            plt.annotate(f'C{cluster_id}', centroid, fontsize=12, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {output_path}")
        else:
            plt.show()
    
    def generate_report(self, classification_df: pd.DataFrame, 
                       clustering_df: pd.DataFrame = None) -> str:
        """
        Generate a comprehensive classification report
        
        Args:
            classification_df: Classification results
            clustering_df: Clustering results (optional)
            
        Returns:
            Report text
        """
        report = []
        report.append("🔍 LOCATOR CLASSIFICATION REPORT")
        report.append("=" * 50)
        
        # Classification summary
        report.append(f"\n📊 CLASSIFICATION SUMMARY")
        report.append(f"Total locators analyzed: {len(classification_df)}")
        report.append(f"Categories found: {classification_df['predicted_category'].nunique()}")
        
        # Category distribution
        report.append(f"\n📈 CATEGORY DISTRIBUTION:")
        category_counts = classification_df['predicted_category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(classification_df)) * 100
            report.append(f"  {category}: {count} ({percentage:.1f}%)")
        
        # High confidence predictions
        report.append(f"\n⭐ HIGH CONFIDENCE PREDICTIONS (>0.7):")
        high_conf = classification_df[classification_df['confidence'] > 0.7]
        report.append(f"  {len(high_conf)} out of {len(classification_df)} ({len(high_conf)/len(classification_df)*100:.1f}%)")
        
        # Examples by category
        report.append(f"\n📝 EXAMPLES BY CATEGORY:")
        for category in category_counts.index[:5]:  # Top 5 categories
            examples = classification_df[classification_df['predicted_category'] == category]['locator'].head(3).tolist()
            report.append(f"  {category}:")
            for example in examples:
                report.append(f"    • {example}")
        
        # Source file analysis
        report.append(f"\n📁 SOURCE FILE ANALYSIS:")
        file_counts = classification_df['source_file'].value_counts()
        for file_name, count in file_counts.head(5).items():
            report.append(f"  {file_name}: {count} locators")
        
        # Weight analysis
        report.append(f"\n⚖️ WEIGHT ANALYSIS:")
        weight_stats = classification_df['weight'].describe()
        report.append(f"  Average weight: {weight_stats['mean']:.2f}")
        report.append(f"  Weight range: {weight_stats['min']:.2f} to {weight_stats['max']:.2f}")
        
        # High weight locators
        high_weight = classification_df[classification_df['weight'] >= 5].sort_values('weight', ascending=False)
        if len(high_weight) > 0:
            report.append(f"\n🎯 HIGH WEIGHT LOCATORS (≥5):")
            for _, row in high_weight.head(5).iterrows():
                report.append(f"  {row['locator']} (weight: {row['weight']}, category: {row['predicted_category']})")
        
        # Clustering summary (if available)
        if clustering_df is not None:
            report.append(f"\n🎯 CLUSTERING SUMMARY:")
            n_clusters = clustering_df['cluster'].nunique()
            report.append(f"Number of clusters: {n_clusters}")
            
            if hasattr(clustering_df, 'attrs') and 'cluster_summaries' in clustering_df.attrs:
                summaries = clustering_df.attrs['cluster_summaries']
                for cluster_id, summary in summaries.items():
                    report.append(f"  Cluster {cluster_id}: {summary['size']} items")
                    report.append(f"    Examples: {', '.join(summary['examples'])}")
                    report.append(f"    Common words: {', '.join(summary['common_words'])}")
        
        return "\\n".join(report)
    
    def save_results(self, classification_df: pd.DataFrame, 
                    clustering_df: pd.DataFrame = None,
                    output_dir: str = "locator_analysis_output") -> None:
        """
        Save all results to files
        
        Args:
            classification_df: Classification results
            clustering_df: Clustering results (optional)
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save classification results
        classification_df.to_csv(output_path / "classification_results.csv", index=False)
        print(f"💾 Classification results saved to: {output_path / 'classification_results.csv'}")
        
        # Save clustering results
        if clustering_df is not None:
            # Save without embeddings for CSV
            clustering_csv = clustering_df.drop('embedding', axis=1)
            clustering_csv.to_csv(output_path / "clustering_results.csv", index=False)
            print(f"💾 Clustering results saved to: {output_path / 'clustering_results.csv'}")
            
            # Save full clustering results with embeddings as pickle
            clustering_df.to_pickle(output_path / "clustering_results.pkl")
            print(f"💾 Full clustering data saved to: {output_path / 'clustering_results.pkl'}")
        
        # Generate and save report
        report = self.generate_report(classification_df, clustering_df)
        with open(output_path / "analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📊 Analysis report saved to: {output_path / 'analysis_report.txt'}")

def main():
    """Main function to run the Brazilian Energy Bill locator analysis"""
    print("🚀 Starting Brazilian Energy Bill Locator Classification Analysis")
    print("=" * 60)
    
    # Initialize classifier
    classifier = LocatorClassifier()
    
    # Load locators
    locators_file = "default_locators.json"
    if not Path(locators_file).exists():
        print(f"❌ Locators file not found: {locators_file}")
        return
    
    print(f"📂 Loading locators from: {locators_file}")
    locators_data = classifier.load_locators(locators_file)
    
    # Extract all locators
    all_locators = classifier.extract_all_locators(locators_data)
    print(f"📊 Extracted {len(all_locators)} total locators")
    
    # Perform classification
    print(f"\\n🔍 Performing semantic classification...")
    classification_results = classifier.classify_locators(all_locators)
    
    # Perform clustering
    print(f"\\n🎯 Performing clustering analysis...")
    clustering_results = classifier.cluster_locators(all_locators, n_clusters=10)
    
    # Visualize clusters
    print(f"\\n📊 Creating visualizations...")
    classifier.visualize_clusters(clustering_results, "locator_analysis_output/clusters_visualization.png")
    
    # Save all results
    print(f"\\n💾 Saving results...")
    classifier.save_results(classification_results, clustering_results)
    
    # Print summary
    print(f"\\n" + "=" * 50)
    print(f"✅ Analysis complete!")
    print(f"📁 Results saved to: locator_analysis_output/")
    print(f"📊 Check analysis_report.txt for detailed insights")
    
    # Print quick summary
    print(f"\\n📈 Quick Summary:")
    print(f"  Total locators: {len(all_locators)}")
    print(f"  Categories identified: {classification_results['predicted_category'].nunique()}")
    print(f"  Clusters formed: {clustering_results['cluster'].nunique()}")
    print(f"  Average confidence: {classification_results['confidence'].mean():.3f}")

if __name__ == "__main__":
    main()
