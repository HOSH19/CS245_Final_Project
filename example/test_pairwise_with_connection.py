# example/test_pairwise_with_connection.py
'''
Test for PairwiseRanker module.
Using a mock LLM to simulate pairwise comparisons.

usage:
$env:PYTHONPATH=".."; python test_pairwise_with_connection.py
'''

import sys
import os
from dotenv import load_dotenv

# 1. Set path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Import
try:
    from GoogleGeminiLLM import GoogleGeminiLLM 
    from websocietysimulator.agent.modules.pairwise_modules import PairwiseRanker
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

load_dotenv()

# ==========================================
# Prepare Test Data 
# ==========================================

# We deliberately put the BEST book (SciFi) at the 3rd position (index 2).
# We put the WORST book (Cooking) at the 1st position (index 0).
# If the ranker works, SciFi should jump to #1.
initial_ranking = ["book_cooking", "book_romance", "book_scifi"] + [f"other_{i}" for i in range(17)]

context = {
    "user_profile": {
        "user_id": "u_test",
        "preferences": "Loves Science Fiction, Space Opera, and Aliens. Dislikes cooking.",
    },
    "item_profiles": [
        {
            "item_id": "book_scifi",
            "title": "Dune",
            "genre": "Science Fiction",
            "description": "A story about spice and sand worms on a desert planet."
        },
        {
            "item_id": "book_cooking",
            "title": "Mastering Pasta",
            "genre": "Cookbook",
            "description": "How to make the best spaghetti."
        },
        {
            "item_id": "book_romance",
            "title": "Love in Paris",
            "genre": "Romance",
            "description": "A romantic story about two lovers meeting in France."
        }
    ]
}
# Fill dummy data
for i in range(17):
    context["item_profiles"].append({"item_id": f"other_{i}", "title": "Unknown", "description": "N/A"})

# ==========================================
# Run Test
# ==========================================
def run_test():
    print("--- Initializing Google Gemini LLM ---")
    
    try:
        llm = GoogleGeminiLLM() 
        print(f"✓ LLM Initialized: {llm.model}")
    except Exception as e:
        print(f"❌ LLM Init Failed: {e}")
        return

    ranker = PairwiseRanker(llm)

    print("\n--- Starting Pairwise Rerank ---")
    print(f"User Prefs: {context['user_profile']['preferences']}")
    print(f"Initial Order (Top 3): {initial_ranking[:3]}")
    print("(Note: 'book_scifi' is currently at position 3. It should move to #1)")
    
    try:
        final_ranking = ranker.rerank(initial_ranking, context)
        
        print("\n--- Final Result ---")
        top_3_ids = final_ranking[:3]
        print(f"Top 3 Item IDs: {top_3_ids}")
        
        # Validation Logic
        if top_3_ids[0] == "book_scifi":
            print("\n✅ SUCCESS: Gemini successfully promoted 'book_scifi' from #3 to #1!")
        elif top_3_ids[0] == "book_cooking":
             print("\n❌ FAILED: The ranking didn't change. 'book_cooking' is still #1.")
        else:
            print(f"\n⚠️  WARNING: Unexpected result. Winner is {top_3_ids[0]}.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()