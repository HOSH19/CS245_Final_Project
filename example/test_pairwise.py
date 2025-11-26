# test_pairwise.py
'''
Test for PairwiseRanker module.
Using a mock LLM to simulate pairwise comparisons.

if error "ModuleNotFoundError: No module named 'websocietysimulator'" in terminal output, run:
$env:PYTHONPATH=".."; python test_pairwise.py
'''
import json
from websocietysimulator.agent.modules.pairwise_modules import PairwiseRanker

class MockLLM:
    def generate(self, prompt):
        """"
        Simulation Logic:
        We assume 'item_2' is the strongest book (originally ranked 3rd).
        As long as 'item_2' appears in the prompt, the side possessing it is the winner.
        """
        
        target_winner_id = "item_2" # This is our predetermined champion

        # 1. Ensure the target is in the Prompt (if this match is unrelated to item_2, return arbitrary result)
        if target_winner_id not in prompt:
            return "A"

        # 2. Split the Prompt to distinguish between Candidate A and Candidate B
        # Prompt structure: ... [Book A] ... (JSON A) ... [Book B] ... (JSON B)
        parts = prompt.split("[Book B]")
        
        part_a_content = parts[0]
        part_b_content = parts[1] if len(parts) > 1 else ""

        # 3. Determine which side item_2 is on
        if target_winner_id in part_a_content:
            return "A" # item_2 is A, so A wins
        elif target_winner_id in part_b_content:
            return "B" # item_2 is B, so B wins
            
        return "A" # Fallback

# --- Prepare Test Data ---

# 1. Create 20 candidate IDs
# Order: item_0 (1st), item_1 (2nd), item_2 (3rd)...
initial_ranking = [f"item_{i}" for i in range(20)]

# 2. Prepare Context (Profile)
#  Each item profile must contain 'item_id' so the MockLLM can see it
context = {
    "user_profile": {"user_id": "u1", "pref": "history"},
    "item_profiles": []
}
print("--- Context ---")
print(context)
for i, item_id in enumerate(initial_ranking):
    context["item_profiles"].append({
        "item_id": item_id,   # <--- This is the keyword MockLLM will look for
        "title": f"Random Title {i}", 
        "description": "Some description..."
    })

# --- Start Test ---
print("--- Test Start ---")
print(f"Initial Top 3 IDs: {initial_ranking[:3]}")
# Expected: ['item_0', 'item_1', 'item_2']
# Currently item_2 is at the end (3rd place)

ranker = PairwiseRanker(MockLLM())
final_ranking = ranker.rerank(initial_ranking, context)

print(f"Final ranking:   {final_ranking}")

# --- Validation ---
# We expect item_2 (since it's the MockLLM's designated strongest) to rush to 1st place
if final_ranking[0] == "item_2":
    print("\n✅ SUCCESS: item_2 successfully became #1! Pairwise logic is working correctly.")
else:
    print(f"\n❌ FAILED: The top item is {final_ranking[0]}, but expected item_2.")