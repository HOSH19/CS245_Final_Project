import itertools
import json

class PairwiseRanker:
    def __init__(self, llm):
        self.llm = llm
        self.K = 3  # Only rerank the top 3 items

    def rerank(self, initial_ranking, context):
        """
        Input:
            initial_ranking: List of 20 item IDs
            context: Dict containing 'user_profile' and 'item_profiles'
        Output:
            final_ranking: List of 20 item IDs (Top 3 refined + Rest 17 original)
        """
        
        # 1. Create lookup table (Item ID -> Profile)
        item_map = {item['item_id']: item for item in context['item_profiles']}
        user_profile = context['user_profile']

        # 2. Split: Top 3 (to be reranked) and Rest (unchanged)
        top_k_ids = initial_ranking[:self.K]
        rest_ids = initial_ranking[self.K:]  # the remaining 4th-20th items

        # 3. Initialize scoreboard (only for Top 3)
        scores = {item_id: 0 for item_id in top_k_ids}

        # 4. Round-Robin pairwise comparison
        pairs = list(itertools.combinations(top_k_ids, 2))

        for id_a, id_b in pairs:
            profile_a = item_map.get(id_a)
            profile_b = item_map.get(id_b)
            
            # Call LLM
            winner_id = self.compare_pair(user_profile, profile_a, profile_b)

            # [DEBUG] Print winner
            winner_name = "A" if winner_id == id_a else "B" if winner_id == id_b else "Tie/Error"
            print(f"[DEBUG] {id_a} vs {id_b} -> Winner: {winner_name}")

            # Scoring
            if winner_id == id_a:
                scores[id_a] += 1
                scores[id_b] -= 1
            elif winner_id == id_b:
                scores[id_b] += 1
                scores[id_a] -= 1

        # [DEBUG] Print final scores
        print(f"[DEBUG] Final Scores: {scores}")
        
        # 5. Sort Top 3 (Stable Sort)
        # Items with higher scores rank higher; ties resolved by original order
        sorted_top_3 = sorted(
            top_k_ids,
            key=lambda item_id: (-scores[item_id], top_k_ids.index(item_id))
        )

        # 6. Concatenate the sorted Top 3 with the rest
        final_ranking = sorted_top_3 + rest_ids

        return final_ranking

    def compare_pair(self, user_profile, item_a, item_b):
        try:
            # Prepare Prompt
            user_str = json.dumps(user_profile, ensure_ascii=False)
            item_a_str = json.dumps(item_a, ensure_ascii=False)
            item_b_str = json.dumps(item_b, ensure_ascii=False)

            prompt = f"""
I need you to act as an expert book recommendation judge.

You are given a User Profile and two Book Candidates (Book A and Book B).
Your task is to analyze the full details of the user and the books to decide which book is a better match.

[User Profile]
{user_str}

[Book A]
{item_a_str}

[Book B]
{item_b_str}

**Task:**
Compare Book A and Book B. Which one is a better recommendation for this user?

**Rules:**
1. Match the user's preferences with the book's attributes.
2. Ignore the order of presentation.
3. Output ONLY 'A' if Book A is better, 'B' if Book B is better.
"""
            # call LLM
            messages = [{"role": "user", "content": prompt}]
            # Call self.llm directly (triggers GoogleGeminiLLM's __call__)
            # response will be a string
            response = self.llm(messages=messages)
            
            # Ensure it is a string and convert to uppercase (GoogleGeminiLLM returns str)
            response = response.strip().upper()

            # get the IDs for further computation
            id_a, id_b = item_a['item_id'], item_b['item_id']

            # Parse response
            if 'A' in response and 'B' not in response: return id_a
            if 'B' in response and 'A' not in response: return id_b
            if response == 'A': return id_a
            if response == 'B': return id_b
            if response.startswith('A'): return id_a
            if response.startswith('B'): return id_b
            
            return None 

        except Exception:
            return None