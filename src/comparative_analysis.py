"""
Enhanced Comparative Analysis Module
Adds accuracy metrics and comparative analysis to the AI Diet & Food Safety Recommender
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderAnalytics:
    """
    Comprehensive analytics for the food safety recommender system
    """
    
    def __init__(self):
        self.metrics = {}
        
    # ========================================
    # 1. MODEL ACCURACY METRICS
    # ========================================
    
    def calculate_model_accuracy(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Calculate comprehensive accuracy metrics for a classification model
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities for ROC-AUC
        model_name : str
            Name of the model for reporting
            
        Returns:
        --------
        dict : Dictionary containing all accuracy metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
                
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def compare_models(self, model_results):
        """
        Compare multiple models side-by-side
        
        Parameters:
        -----------
        model_results : list of dict
            List of metric dictionaries from calculate_model_accuracy
            
        Returns:
        --------
        pd.DataFrame : Comparison table
        """
        comparison_df = pd.DataFrame(model_results)
        comparison_df = comparison_df.drop('confusion_matrix', axis=1, errors='ignore')
        comparison_df = comparison_df.round(4)
        
        # Sort by F1 score
        comparison_df = comparison_df.sort_values('f1_weighted', ascending=False)
        
        return comparison_df
    
    # ========================================
    # 2. RECOMMENDATION QUALITY METRICS
    # ========================================
    
    def calculate_ndcg(self, recommended_scores, k=5):
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        Measures ranking quality
        
        Parameters:
        -----------
        recommended_scores : array-like
            Recommendation scores in ranked order
        k : int
            Number of top items to consider
            
        Returns:
        --------
        float : NDCG@K score
        """
        recommended_scores = np.array(recommended_scores[:k])
        
        # DCG
        dcg = np.sum(recommended_scores / np.log2(np.arange(2, len(recommended_scores) + 2)))
        
        # IDCG (ideal DCG with perfect ranking)
        ideal_scores = np.sort(recommended_scores)[::-1]
        idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_diversity_metrics(self, recommendations_df):
        """
        Calculate diversity metrics for recommendations
        
        Parameters:
        -----------
        recommendations_df : pd.DataFrame
            DataFrame with recommendations containing product categories
            
        Returns:
        --------
        dict : Diversity metrics
        """
        metrics = {
            'unique_products': recommendations_df['product_name'].nunique(),
            'total_recommendations': len(recommendations_df),
            'category_coverage': recommendations_df['PROD_CAT'].nunique() if 'PROD_CAT' in recommendations_df else 0,
            'avg_recommendation_score': recommendations_df['recommendation_score'].mean() if 'recommendation_score' in recommendations_df else 0,
            'score_std': recommendations_df['recommendation_score'].std() if 'recommendation_score' in recommendations_df else 0,
        }
        
        return metrics
    
    def calculate_personalization_effectiveness(self, recommendations_df, user_profiles_df):
        """
        Measure how well recommendations match user preferences
        
        Parameters:
        -----------
        recommendations_df : pd.DataFrame
            User recommendations
        user_profiles_df : pd.DataFrame
            User profiles with preferences
            
        Returns:
        --------
        dict : Personalization metrics
        """
        metrics = {
            'users_with_recommendations': recommendations_df['user_id'].nunique(),
            'total_users': len(user_profiles_df),
            'coverage_rate': recommendations_df['user_id'].nunique() / len(user_profiles_df),
            'avg_recommendations_per_user': recommendations_df.groupby('user_id').size().mean(),
        }
        
        return metrics
    
    # ========================================
    # 3. COMPARATIVE ANALYSIS
    # ========================================
    
    def compare_by_diet_type(self, recommendations_df):
        """
        Compare recommendations across different diet types
        
        Parameters:
        -----------
        recommendations_df : pd.DataFrame
            Recommendations with diet_type column
            
        Returns:
        --------
        pd.DataFrame : Comparison by diet type
        """
        if 'diet_type' not in recommendations_df.columns:
            return None
            
        comparison = recommendations_df.groupby('diet_type').agg({
            'recommendation_score': ['mean', 'std', 'min', 'max'],
            'safety_score': ['mean', 'std'],
            'health_score': ['mean', 'std'],
            'product_name': 'count'
        }).round(3)
        
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        comparison = comparison.rename(columns={'product_name_count': 'num_recommendations'})
        
        return comparison
    
    def compare_by_health_goal(self, recommendations_df):
        """
        Compare recommendations across different health goals
        """
        if 'goal' not in recommendations_df.columns:
            return None
            
        comparison = recommendations_df.groupby('goal').agg({
            'recommendation_score': ['mean', 'std'],
            'safety_score': 'mean',
            'health_score': 'mean',
            'energy-kcal_100g': ['mean', 'std'],
            'proteins_100g': ['mean', 'std'],
            'product_name': 'count'
        }).round(3)
        
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        comparison = comparison.rename(columns={'product_name_count': 'num_recommendations'})
        
        return comparison
    
    def compare_by_safety_risk_level(self, products_df):
        """
        Compare products by safety risk categories
        """
        if 'safety_score' not in products_df.columns:
            return None
            
        # Create risk categories
        products_df['risk_category'] = pd.cut(
            products_df['safety_score'],
            bins=[0, 0.5, 0.7, 0.85, 1.0],
            labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Safe']
        )
        
        comparison = products_df.groupby('risk_category').agg({
            'safety_score': ['mean', 'count'],
            'health_score': 'mean',
            'PROD_CAT': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
        }).round(3)
        
        comparison.columns = ['_'.join(col).strip() if col[1] else col[0] for col in comparison.columns.values]
        comparison = comparison.rename(columns={'safety_score_count': 'num_products', 'PROD_CAT_<lambda>': 'most_common_category'})
        
        return comparison
    
    def compare_by_nutritional_quality(self, products_df):
        """
        Compare products by nutritional quality
        """
        if 'health_score' not in products_df.columns:
            return None
            
        # Create health categories
        products_df['health_category'] = pd.cut(
            products_df['health_score'],
            bins=[0, 0.5, 0.7, 0.85, 1.0],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        comparison = products_df.groupby('health_category').agg({
            'health_score': ['mean', 'count'],
            'safety_score': 'mean',
            'energy-kcal_100g': 'mean',
            'proteins_100g': 'mean',
            'sugars_100g': 'mean',
            'fiber_100g': 'mean'
        }).round(3)
        
        comparison.columns = ['_'.join(col).strip() if col[1] else col[0] for col in comparison.columns.values]
        comparison = comparison.rename(columns={'health_score_count': 'num_products'})
        
        return comparison
    
    def compare_by_gender(self, recommendations_df):
        """
        Compare recommendations by user gender
        """
        if 'gender' not in recommendations_df.columns:
            return None
            
        comparison = recommendations_df.groupby('gender').agg({
            'recommendation_score': ['mean', 'std'],
            'safety_score': 'mean',
            'health_score': 'mean',
            'product_name': 'count'
        }).round(3)
        
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        comparison = comparison.rename(columns={'product_name_count': 'num_recommendations'})
        
        return comparison

    def compare_by_activity_level(self, recommendations_df):
        """
        Compare recommendations by user activity level
        """
        if 'activity_level' not in recommendations_df.columns:
            return None
            
        comparison = recommendations_df.groupby('activity_level').agg({
            'recommendation_score': ['mean', 'std'],
            'safety_score': 'mean',
            'health_score': 'mean',
            'energy-kcal_100g': 'mean',
            'product_name': 'count'
        }).round(3)
        
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        comparison = comparison.rename(columns={'product_name_count': 'num_recommendations'})
        
        return comparison
    
    # ========================================
    # 4. VISUALIZATION
    # ========================================
    
    def plot_model_comparison(self, comparison_df, save_path=None):
        """
        Visualize model comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        titles = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx // 2, idx % 2]
            if metric in comparison_df.columns:
                comparison_df.plot(x='model_name', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
                ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
                ax.set_ylabel(title)
                ax.set_xlabel('Model')
                ax.set_ylim([0, 1])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparative_analysis(self, comparison_df, title, save_path=None):
        """
        Generic comparative analysis visualization
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot grouped bar chart for score columns
        score_cols = [col for col in comparison_df.columns if 'mean' in col and 'score' in col]
        
        if score_cols:
            comparison_df[score_cols].plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_xlabel('Category')
            ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ========================================
    # 5. COMPREHENSIVE REPORT
    # ========================================
    
    def generate_comprehensive_report(self, model_metrics, recommendations_df, products_df, user_profiles_df):
        """
        Generate a comprehensive analysis report
        
        Returns:
        --------
        dict : Complete analysis report
        """
        report = {
            'model_performance': model_metrics,
            'recommendation_quality': self.calculate_diversity_metrics(recommendations_df),
            'personalization': self.calculate_personalization_effectiveness(recommendations_df, user_profiles_df),
            'diet_type_comparison': self.compare_by_diet_type(recommendations_df),
            'health_goal_comparison': self.compare_by_health_goal(recommendations_df),
            'safety_risk_comparison': self.compare_by_safety_risk_level(products_df),
            'nutritional_quality_comparison': self.compare_by_nutritional_quality(products_df),
            'gender_comparison': self.compare_by_gender(recommendations_df),
            'activity_level_comparison': self.compare_by_activity_level(recommendations_df)
        }
        
        return report
    
    def print_report_summary(self, report):
        """
        Print a formatted summary of the analysis report
        """
        print("=" * 80)
        print("📊 COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        
        print("\n🎯 MODEL PERFORMANCE")
        print("-" * 80)
        if isinstance(report['model_performance'], dict):
            for key, value in report['model_performance'].items():
                if key != 'confusion_matrix':
                    print(f"  {key}: {value}")
        
        print("\n📈 RECOMMENDATION QUALITY")
        print("-" * 80)
        for key, value in report['recommendation_quality'].items():
            print(f"  {key}: {value}")
        
        print("\n👤 PERSONALIZATION EFFECTIVENESS")
        print("-" * 80)
        for key, value in report['personalization'].items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    print("Enhanced Comparative Analysis Module")
    print("=" * 80)
    print("\nThis module provides:")
    print("  ✓ Model accuracy metrics (accuracy, precision, recall, F1)")
    print("  ✓ Recommendation quality metrics (NDCG, diversity)")
    print("  ✓ Comparative analysis by diet type, health goal, safety risk")
    print("  ✓ Visualization tools for all comparisons")
    print("  ✓ Comprehensive reporting")
    print("\n" + "=" * 80)
