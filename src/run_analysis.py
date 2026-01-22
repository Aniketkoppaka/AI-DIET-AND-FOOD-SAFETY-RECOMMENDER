"""
Generate recommendations and run comprehensive comparative analysis
"""
import pandas as pd
import numpy as np
import os
import sys
from .comparative_analysis import RecommenderAnalytics

# Fix for plotting without display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("="*80)
    print("🚀 STARTING AI DIET & FOOD SAFETY COMPARATIVE ANALYSIS")
    print("="*80)

    # 2. SIMULATE RECOMMENDATIONS (Simplified Logic)
    print("\n🤖 Generating synthetic data for analysis demo...")
    
    # Create synthetic products
    n_products = 1000
    print(f"   Creating {n_products} synthetic products...")
    products_df = pd.DataFrame({
        'product_name': [f'Product_{i}' for i in range(n_products)],
        'PROD_CAT': np.random.choice(['Vegetables', 'Fruits', 'Dairy', 'Grains', 'Meat', 'Seafood'], n_products),
        'safety_score': np.random.uniform(0.5, 1.0, n_products),
        'health_score': np.random.uniform(0.4, 1.0, n_products),
        'energy-kcal_100g': np.random.uniform(50, 500, n_products),
        'proteins_100g': np.random.uniform(0, 30, n_products),
        'sugars_100g': np.random.uniform(0, 20, n_products),
        'fiber_100g': np.random.uniform(0, 15, n_products)
    })
    products_df['recommendation_score'] = (0.6 * products_df['safety_score'] + 0.4 * products_df['health_score'])
    
    # Create synthetic users
    n_users = 100
    print(f"   Creating {n_users} synthetic users...")
    users_df = pd.DataFrame({
        'user_id': [f'User_{i}' for i in range(n_users)],
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_users),
        'activity_level': np.random.choice(['Sedentary', 'Moderate', 'Active'], n_users),
        'diet_type': np.random.choice(['Vegan', 'Vegetarian', 'Keto', 'Regular'], n_users),
        'goal': np.random.choice(['Weight Loss', 'Maintenance', 'Muscle Gain'], n_users)
    })
    
    recommendations = []
    
    for _, user in users_df.iterrows():
        # Pick 5 random products as recs
        user_recs = products_df.sample(5)
        for _, rec in user_recs.iterrows():
            recommendations.append({
                'user_id': user['user_id'],
                'gender': user['gender'],
                'diet_type': user['diet_type'],
                'goal': user['goal'],
                'activity_level': user['activity_level'],
                'product_name': rec['product_name'],
                'safety_score': rec['safety_score'],
                'health_score': rec['health_score'],
                'recommendation_score': rec['recommendation_score'],
                'PROD_CAT': rec['PROD_CAT'],
                'energy-kcal_100g': rec['energy-kcal_100g'],
                'proteins_100g': rec['proteins_100g'],
                'fiber_100g': rec['fiber_100g']
            })
            
    recommendations_df = pd.DataFrame(recommendations)
    print(f"   ✅ Generated {len(recommendations_df)} recommendations")

    # 3. RUN ANALYSIS
    print("\n📊 Running Comparative Analysis...")
    analytics = RecommenderAnalytics()
    
    # Reports
    print("\n   --- Demographics Analysis ---")
    
    print("\n   1. By Diet Type:")
    diet_comp = analytics.compare_by_diet_type(recommendations_df)
    print(diet_comp)
    
    print("\n   2. By Health Goal:")
    goal_comp = analytics.compare_by_health_goal(recommendations_df)
    print(goal_comp)
    
    if analytics.compare_by_gender(recommendations_df) is not None:
        print("\n   3. By Gender:")
        print(analytics.compare_by_gender(recommendations_df))
        
    if analytics.compare_by_activity_level(recommendations_df) is not None:
        print("\n   4. By Activity Level:")
        print(analytics.compare_by_activity_level(recommendations_df))
        
    print("\n   --- Product Analysis ---")
    
    print("\n   5. By Safety Risk Level (All Products):")
    risk_comp = analytics.compare_by_safety_risk_level(products_df)
    print(risk_comp)
    
    print("\n   6. By Nutritional Quality (All Products):")
    nutri_comp = analytics.compare_by_nutritional_quality(products_df)
    print(nutri_comp)
    
    # 4. VISUALIZATIONS
    print("\n🎨 Generating Visualizations...")
    
    if diet_comp is not None:
        analytics.plot_comparative_analysis(
            diet_comp, 
            'Recommendation Scores by Diet Type', 
            save_path='outputs/analysis_diet_type.png'
        )
        print("   Saved analysis_diet_type.png")
        
    if goal_comp is not None:
        analytics.plot_comparative_analysis(
            goal_comp, 
            'Scores by Health Goal', 
            save_path='outputs/analysis_health_goal.png'
        )
        print("   Saved analysis_health_goal.png")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
