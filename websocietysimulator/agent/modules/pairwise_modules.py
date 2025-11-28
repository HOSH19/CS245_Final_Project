import itertools
import json

class PairwiseRanker:
    def __init__(self, llm):
        self.llm = llm
        self.K = 5  # Only rerank the top 5 items
        # Since we use "King of the Hill", this only costs 4 API calls per task.

    def rerank(self, initial_ranking, context):
        """
        Input:
            initial_ranking: List of 20 item IDs
            context: Dict containing 'user_profile' and 'item_profiles'
        Output:
            final_ranking: List of 20 item IDs (Winner moved to #1 + Rest)
        """
        
        # 1. Create lookup table (Item ID -> Profile)
        item_map = {item['item_id']: item for item in context['item_profiles']}
        user_profile = context['user_profile']

        # 2. Split: Top K (Participants) and Rest (Spectators)
        candidates = initial_ranking[:self.K]
        rest = initial_ranking[self.K:]

        # Safety check: if less than 2 items, no comparison needed
        if len(candidates) < 2:
            return initial_ranking

        # 3. King of the Hill Strategy (Linear Scan)
        # We assume the first item is the current "King" (Winner).
        current_winner_id = candidates[0]
        
        print(f"\n[Pairwise] Starting 'King of the Hill' scan on Top {self.K} candidates...")

        # Iterate through the challengers (Item 2 to Item 5)
        for challenger_id in candidates[1:]:
            
            # Retrieve profiles
            profile_winner = item_map.get(current_winner_id)
            profile_challenger = item_map.get(challenger_id)
            
            # Skip if profile missing (should not happen if data is clean)
            if not profile_winner or not profile_challenger:
                continue

            # Compare: Current King (Item A) vs Challenger (Item B)
            winner_result = self.compare_pair(user_profile, profile_winner, profile_challenger)
            
            # Log the fight result
            winner_label = "Challenger" if winner_result == challenger_id else "Current King"
            print(f"[DEBUG] {current_winner_id} (King) vs {challenger_id} (Challenger) -> Winner: {winner_label}")

            # Update the King if the challenger wins
            if winner_result == challenger_id:
                current_winner_id = challenger_id
            else:
                # Current winner defends the title, continues to next round
                pass

        print(f"[Pairwise] Final Winner identified: {current_winner_id}")

        # 4. Reconstruct the list
        # Strategy: Move the ultimate winner to Rank 1.
        # Keep the others in their original relative order to preserve Pointwise signals.
        
        others = [x for x in candidates if x != current_winner_id]
        final_top_k = [current_winner_id] + others
        
        # 5. Concatenate with the rest of the list
        final_ranking = final_top_k + rest

        final_ranking = [str(item) for item in final_ranking]
        if len(final_ranking) != len(initial_ranking):
            print(f"[ERROR] List length changed! Original: {len(initial_ranking)}, New: {len(final_ranking)}")
            # if the lengths don't match, return the initial ranking
            return initial_ranking
        
        return final_ranking

    def compare_pair(self, user_profile, item_a, item_b):
        try:
            # Prepare Prompt
            user_str = json.dumps(user_profile, ensure_ascii=False)
            item_a_str = json.dumps(item_a, ensure_ascii=False)
            item_b_str = json.dumps(item_b, ensure_ascii=False)

            # >>>>>>>>> STRICTER PROMPT FOR SIGNIFICANT DIFFERENCE >>>>>>>>>
            prompt = f"""
Role: You are a strict personal librarian and recommendation judge.
Your goal is to determine if a new book candidate is **significantly better** than the current best option for this specific user.

[User Profile]
{user_str}

[Candidate B (The Challenger)]
{item_b_str}

[Candidate A (The Current Champion)]
{item_a_str}

**Task:**
Compare Candidate A and Candidate B strictly based on the user's taste history.
You must decide if Candidate B provides a specific value that Candidate A misses.

**Strict Evaluation Rules:**
1. **Threshold:** Only pick Candidate B if it is a **SIGNIFICANTLY better fit** for the user's specific sub-genres or mood than Candidate A.
2. **Tie-Breaker:** If both books are equally good fits, or if the difference is subjective, **you MUST stick with Candidate A**.
3. **Niche over Popularity:** Do not pick B just because it is popular. It must match the user's unique history.

**Output:**
Reasoning: [Explain why B is/is not significantly better than A]
Winner: [A or B]
"""
            # <<<<<<<<< PROMPT END <<<<<<<<<

            # call LLM
            messages = [{"role": "user", "content": prompt}]
            response = self.llm(messages=messages)

            # Debug: look what LLM responded
            print(f"[LLM Thinking]: {response}")
            
            # Ensure it is a string and convert to uppercase (GoogleGeminiLLM returns str)
            response = response.strip().upper()

            # get the IDs for further computation
            id_a, id_b = item_a['item_id'], item_b['item_id']

            # ===== ROBUST PARSING =====
            # 1. Precise match (Explicit Match)
            if "WINNER: B" in response or "WINNER: [B]" in response:
                return id_b
            if "WINNER: A" in response or "WINNER: [A]" in response:
                return id_a
            # 2. Check the end of the string (End of String Check)
            # Since the prompt requests the Winner on the last line, checking the last character is usually accurate.
            clean_response = response.replace(".", "").strip() # remove periods and whitespace
            if clean_response.endswith("WINNER: B") or clean_response.endswith(" B"):
                return id_b
            if clean_response.endswith("WINNER: A") or clean_response.endswith(" A"):
                return id_a
            # 3. If the above fail, use loose checking (but be careful of noise in Reasoning)
            # Here we only check the "last line" of text to avoid reading A/B within the Reasoning section.
            last_line = response.split('\n')[-1] # Look at the last line only
            if "B" in last_line and "A" not in last_line: return id_b
            if "A" in last_line and "B" not in last_line: return id_a
            
            # Default: If parsing fails, maintain original ruling (King Wins)
            return id_a 
        
        except Exception:
            return None