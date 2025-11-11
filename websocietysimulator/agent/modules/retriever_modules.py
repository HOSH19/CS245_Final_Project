# User profile retrieval utilities for reasoning modules.

import json
import os
from typing import Dict, Optional, Union, Callable

from ...llm import LLMBase


class UserProfileRetriever:
    """
    Loads raw user records from `user.json` and optionally formats them with
    an LLM into a compact profile summary JSON.

    Parameters
    ----------
    data_dir : str
        Directory containing the processed Yelp dataset files.
    llm : LLMBase
        LLM client used to transform raw user data into a profile summary.
    cache : bool
        If True, caches raw user data and generated profiles in-memory.
    profile_prompt_builder : Optional[Callable[[Dict], str]]
        Custom callable to produce the LLM prompt from a raw user record.
        If None, a default prompt is used.
    """

    def __init__(
        self,
        data_dir: str,
        llm: LLMBase,
        cache: bool = True,
        profile_prompt_builder: Optional[Callable[[Dict], str]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.llm = llm
        self.cache = cache
        self.profile_prompt_builder = profile_prompt_builder

        self._user_index: Optional[Dict[str, Dict]] = None
        self._profile_cache: Dict[str, str] = {}

        if cache:
            self._user_index = self._load_all_users()

    def __call__(self, user_id: str) -> Optional[str]:
        if not user_id:
            return None

        if user_id in self._profile_cache:
            return self._profile_cache[user_id]

        raw_profile = self._get_raw_profile(user_id)
        if not raw_profile:
            return None

        formatted_profile = self._format_profile_with_llm(raw_profile)

        if self.cache and formatted_profile:
            self._profile_cache[user_id] = formatted_profile

        return formatted_profile

    def _get_raw_profile(self, user_id: str) -> Optional[Dict]:
        if self._user_index is not None:
            return self._user_index.get(user_id)
        return self._scan_file_for_user(user_id)

    def _load_all_users(self) -> Dict[str, Dict]:
        user_path = os.path.join(self.data_dir, "user.json")
        user_index: Dict[str, Dict] = {}

        with open(user_path, "r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                user_index[record["user_id"]] = record

        return user_index

    def _scan_file_for_user(self, user_id: str) -> Optional[Dict]:
        user_path = os.path.join(self.data_dir, "user.json")

        with open(user_path, "r", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("user_id") == user_id:
                    return record
        return None

    def _format_profile_with_llm(self, raw_profile: Dict) -> Optional[str]:
        if not raw_profile:
            return None

        prompt = (
            self.profile_prompt_builder(raw_profile)
            if self.profile_prompt_builder
            else self._default_profile_prompt(raw_profile)
        )

        response = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.2)

        if not response:
            return None

        return response.strip()

    @staticmethod
    def _default_profile_prompt(raw_profile: Dict) -> str:
        pretty_json = json.dumps(raw_profile, ensure_ascii=False, indent=2)
        return f"""You are given a Yelp user's raw data in JSON format:

{pretty_json}

Summarize this user into a concise JSON object with the following fields:
- "user_id": copy from the raw data.
- "lifetime_stats": highlight key counts or averages (stars, compliments, etc.).
- "preferences": describe cuisine, business types, price-level, or ambiance preferences inferred from the data.
- "review_style": mention tendencies like leniency/harshness, verbosity, use of tips, etc.
- "notable_attributes": any other interesting signals (elite year, fans, compliments).

Respond with JSON only, no additional prose.
"""

