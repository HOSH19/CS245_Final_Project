# pairwise.py

"""
Pairwise reranking module for recommendation tasks

This module performs:
1. Take top-K initial ranking
2. Do pairwise comparison using LLM
3. Compute Copeland scores
4. Tie-break using original initial ranking
"""

class PairwiseReranker:
    def __init__(self, llm, K=5):
        """
        llm: an object with method compare(A, B) -> returns winner item_id
        K: number of top candidates to rerank
        """
        self.llm = llm
        self.K = K

    def compare_pair(self, itemA, itemB):
        """
        Call LLM to decide preferred item between A and B.
        llm.compare() must be implemented in your agent.

        Expected output: itemA or itemB
        """
        try:
            winner = self.llm.compare(itemA, itemB)
            return winner
        except Exception as e:
            print(f"[ERROR] LLM comparison failed for ({itemA}, {itemB}) : {e}")
            # fallback: return original order (tie-break safe)
            return itemA

    def rerank(self, initial_ranking):
        """
        Perform pairwise reranking using Copeland score.

        Input:
            initial_ranking: list of item ids
        Output:
            improved_ranking: list
        """

        # 1. Take top-K
        candidates = initial_ranking[: self.K]

        # Initialize Copeland scores
        scores = {c: 0 for c in candidates}
        pairwise_results = {}

        # 2. Round-robin pairwise compare
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                A = candidates[i]
                B = candidates[j]

                winner = self.compare_pair(A, B)
                pairwise_results[(A, B)] = winner

                # Copeland scoring
                if winner == A:
                    scores[A] += 1
                    scores[B] -= 1
                else:
                    scores[B] += 1
                    scores[A] -= 1

        # 3. Final sorting (Copeland score + tie-breaker)
        improved_ranking = sorted(
            candidates,
            key=lambda c: (-scores[c], initial_ranking.index(c))
        )

        return improved_ranking, scores, pairwise_results
