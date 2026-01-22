"""
AI Diet & Food Safety Recommender - Demo Example
This shows what inputs users provide and what outputs they receive
"""

# ============================================
# EXAMPLE USER INPUT
# ============================================

user_profile = {
    "user_id": "USER_001",
    "age": 28,
    "gender": "Female",
    "country": "USA",
    
    # Dietary Preferences
    "diet_type": "Vegetarian",  # Options: Vegan, Vegetarian, Keto, Regular
    "fav_cuisine": "Mediterranean",
    "preferred_meal_type": "Dinner",
    
    # Health Information
    "goal": "Weight Loss",  # Options: Weight Loss, Muscle Gain, Maintenance
    "allergies": "Nuts, Dairy",
    "health_condition": "None",
    "activity_level": "Moderate",  # Options: Sedentary, Moderate, Active
    
    # Nutritional Targets
    "daily_calorie_target": 1800,
    "protein_target_g": 60,
    "carb_target_g": 200,
    "fat_target_g": 50
}

print("=" * 70)
print("🍽️  AI DIET & FOOD SAFETY RECOMMENDER - DEMO")
print("=" * 70)
print("\n📥 USER INPUT PROFILE:")
print("-" * 70)
for key, value in user_profile.items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

# ============================================
# EXAMPLE SYSTEM OUTPUT
# ============================================

recommendations = [
    {
        "rank": 1,
        "product_name": "Organic Quinoa Salad Bowl",
        "safety_score": 0.92,  # Higher = Safer (0-1 scale)
        "health_score": 0.88,  # Higher = Healthier (0-1 scale)
        "recommendation_score": 0.90,  # Combined score (60% safety + 40% health)
        "calories_per_100g": 120,
        "protein_g": 4.5,
        "carbs_g": 21,
        "fat_g": 2.0,
        "fiber_g": 3.5,
        "match_reason": "Vegetarian, Mediterranean cuisine, Low calorie for weight loss"
    },
    {
        "rank": 2,
        "product_name": "Mediterranean Chickpea Stew",
        "safety_score": 0.89,
        "health_score": 0.85,
        "recommendation_score": 0.87,
        "calories_per_100g": 135,
        "protein_g": 6.2,
        "carbs_g": 22,
        "fat_g": 3.1,
        "fiber_g": 5.0,
        "match_reason": "Vegetarian, Mediterranean cuisine, High protein & fiber"
    },
    {
        "rank": 3,
        "product_name": "Grilled Vegetable Medley",
        "safety_score": 0.91,
        "health_score": 0.82,
        "recommendation_score": 0.87,
        "calories_per_100g": 85,
        "protein_g": 2.8,
        "carbs_g": 12,
        "fat_g": 3.5,
        "fiber_g": 4.2,
        "match_reason": "Vegetarian, Mediterranean style, Very low calorie"
    },
    {
        "rank": 4,
        "product_name": "Lentil & Spinach Curry",
        "safety_score": 0.86,
        "health_score": 0.84,
        "recommendation_score": 0.85,
        "calories_per_100g": 145,
        "protein_g": 7.5,
        "carbs_g": 20,
        "fat_g": 4.0,
        "fiber_g": 6.0,
        "match_reason": "Vegetarian, High protein, Excellent fiber content"
    },
    {
        "rank": 5,
        "product_name": "Greek-Style Stuffed Peppers",
        "safety_score": 0.88,
        "health_score": 0.80,
        "recommendation_score": 0.85,
        "calories_per_100g": 110,
        "protein_g": 5.0,
        "carbs_g": 15,
        "fat_g": 3.2,
        "fiber_g": 3.8,
        "match_reason": "Vegetarian, Mediterranean cuisine, Balanced macros"
    }
]

print("\n" + "=" * 70)
print("📤 PERSONALIZED FOOD RECOMMENDATIONS")
print("=" * 70)

for rec in recommendations:
    print(f"\n🏆 RANK #{rec['rank']}: {rec['product_name']}")
    print(f"   {'─' * 66}")
    print(f"   📊 Overall Score: {rec['recommendation_score']:.2f}/1.00")
    print(f"   🛡️  Safety Score:  {rec['safety_score']:.2f}/1.00 (Low contamination risk)")
    print(f"   💚 Health Score:  {rec['health_score']:.2f}/1.00 (Nutritionally balanced)")
    print(f"\n   📋 Nutrition per 100g:")
    print(f"      • Calories: {rec['calories_per_100g']} kcal")
    print(f"      • Protein:  {rec['protein_g']}g")
    print(f"      • Carbs:    {rec['carbs_g']}g")
    print(f"      • Fat:      {rec['fat_g']}g")
    print(f"      • Fiber:    {rec['fiber_g']}g")
    print(f"\n   ✨ Why recommended: {rec['match_reason']}")

print("\n" + "=" * 70)
print("📈 ADDITIONAL INSIGHTS")
print("=" * 70)
print("""
✓ All recommendations are:
  • Vegetarian-friendly (no meat/fish)
  • Free from your allergens (nuts, dairy)
  • Mediterranean cuisine style
  • Optimized for weight loss goals
  • Scored for both safety and nutrition

✓ Safety scores based on 30,000+ EU food contamination alerts
✓ Health scores based on 400,000+ nutritional profiles
✓ Personalized to your activity level and calorie targets
""")

print("=" * 70)
print("🎯 NEXT STEPS:")
print("=" * 70)
print("""
1. Review your top recommendations
2. Adjust your profile if needed (change diet type, allergies, goals)
3. Get new recommendations tailored to your updated preferences
4. View visualizations comparing safety vs health across food categories
""")
print("=" * 70)
