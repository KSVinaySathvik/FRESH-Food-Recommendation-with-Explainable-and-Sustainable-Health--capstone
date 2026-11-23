"""Label (recipe, user profile) pairs with Llama 3.3 70B to create data/distilled_labels.csv."""
import pandas as pd
import json
import random
from groq import Groq
import os

API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=API_KEY)

print("Loading Metadata...")
with open("data/claude_recipes_metadata_FINAL.json", "r") as f:
    raw_data = json.load(f)

# Handle List vs. Dictionary format
if isinstance(raw_data, list):
    print(f"   Detected LIST format ({len(raw_data)} recipes)")
    all_recipes = raw_data
else:
    print(f"   Detected DICT format ({len(raw_data)} recipes)")
    all_recipes = list(raw_data.values())

data_rows = []

print("Distilling Knowledge from LLM (Teacher)...")

# Generate 50 high-quality training examples
for i in range(50):
    # 1. Random User Persona
    bmi = random.choice([18.0, 22.0, 28.0])
    pantry_mock = random.sample(["onion", "tomato", "rice", "chicken", "milk", "eggs", "potato", "ginger"], 3)
    diet = random.choice(["Vegan", "Non-Veg", "Gluten-Free"])

    # 2. Random Recipe
    recipe = random.choice(all_recipes)

    rid = str(recipe.get("recipe_id") or recipe.get("id") or str(i))

    title = recipe.get("title", "Unknown Recipe")
    ingredients = recipe.get("cleaned_ingredients", [])

    # 3. ask the teacher (llm)
    prompt = f"""
    Act as an expert nutritionist recommender. Rate the match (0.0 to 1.0).

    USER: BMI {bmi}, Diet {diet}, Pantry {pantry_mock}
    RECIPE: {title} (Ingredients: {ingredients})

    Logic:
    - If diet conflict (Vegan vs Meat) -> Score 0.0
    - If pantry matches well -> Boost score.
    - If BMI high and recipe unhealthy -> Lower score.

    OUTPUT JSON ONLY: {{"score": float}}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        score = json.loads(completion.choices[0].message.content)["score"]

        # 4. Save the "Smart Label"
        data_rows.append({
            "recipe_id": rid, # Now guaranteed to be a string
            "user_bmi": bmi,
            "user_pantry_count": len(pantry_mock),
            "label_score": score
        })
        print(f"[{i+1}/50] {title[:30]}... -> Score: {score}")

    except Exception as e:
        print(f"Error on row {i}: {e}")

# Save dataset
if data_rows:
    # Ensure pandas treats the ID column as strings
    df = pd.DataFrame(data_rows)
    df['recipe_id'] = df['recipe_id'].astype(str)
    df.to_csv("data/distilled_labels.csv", index=False)
    print("\nSuccess! Created 'data/distilled_labels.csv' with STRING IDs.")
else:
    print("\nError: Failed to generate data. Check API Key or connection.")
