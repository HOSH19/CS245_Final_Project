import os
from typing import Dict, Any, List
from itertools import islice
from websocietysimulator.agent.modules.schemafitter_modules import SchemaFitterIO
from websocietysimulator.tools.interaction_tool import InteractionTool
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM
from example.GoogleGeminiLLM import GoogleGeminiLLM


def _take_first_n(lst, n):
    return list(islice(lst, n)) if n is not None else lst


class UserProfileBuilder:
    def __init__(self, llm, interaction_tool: InteractionTool):
        self.llm = llm
        self.tool = interaction_tool
        self.schema_fitter = SchemaFitterIO(llm)

    # ---------- single user (unchanged except for max_reviews) ----------
    def build_profile(
        self,
        user_id: str,
        schema: Dict[str, Any],
        max_reviews: int | None = 50,
    ):
        """
        Build a profile for ONE user.

        max_reviews: cap the number of reviews we pass to the LLM (None = no cap).
        """
        user_record = self.tool.get_user(user_id)
        user_reviews = self.tool.get_reviews(user_id=user_id)

        # Limit reviews to avoid huge prompts
        user_reviews = _take_first_n(user_reviews, max_reviews)

        items = {
            "user_id": user_id,
            "user": user_record,
            "reviews": user_reviews,
        }

        structured_profile = self.schema_fitter(schema=schema, items=items)
        return structured_profile

    def build_profiles(
        self,
        user_ids: List[str],
        schema: Dict[str, Any],
        max_reviews: int | None = 50,
        batch_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Build profiles for MANY users.

        Returns:
            dict mapping user_id -> profile object
        """
        results: Dict[str, Any] = {}

        # Simple batching over user_ids
        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i : i + batch_size]

            # Prepare items for this batch
            batch_items = []
            for uid in batch:
                user_record = self.tool.get_user(uid)
                user_reviews = self.tool.get_reviews(user_id=uid)
                user_reviews = _take_first_n(user_reviews, max_reviews)

                batch_items.append(
                    {
                        "user_id": uid,
                        "user": user_record,
                        "reviews": user_reviews,
                    }
                )

            # Call the LLM ONCE for this batch
            batch_profiles = self.schema_fitter(schema=schema, items=batch_items)

            for prof in batch_profiles:
                uid = prof.get("user_id")
                if uid is not None:
                    results[uid] = prof

        return results

def main():
    interaction_tool = InteractionTool(data_dir="./example/dataset")
    google_api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDH08u6IVQXb8yGIzQpHFJ6ba-whu-5lhI')
    llm = GoogleGeminiLLM(
        api_key=google_api_key,
        model="models/gemini-1.5-flash-latest"
    )
    UserProfileBuilder(llm, interaction_tool)

    profile_builder = UserProfileBuilder(llm, interaction_tool)

    schema = {
        "user_id": "string",
        "top_genres": "list[str]",
        "tone_preference": "string",
        "favorite_themes": "list[str]",
        "writing_style_preference": "string",
    }

    profile = profile_builder.build_profile("e5349f9232db517fdfb473641811ac06", schema)
    print(profile)

if __name__ == "__main__":
    main()
