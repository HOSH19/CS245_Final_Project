# """
# Profile module implementations used by `EnhancedRecommendationAgentBase`.

# The profile module is invoked after context gathering and before memory
# integration. It receives the raw `user_profile` and `user_reviews` fetched
# through `InteractionTool` and is expected to return a structured persona block
# that can be injected back into the agent context.
# """

# from __future__ import annotations

# import json
# import logging
# import ast
# from typing import Any, Dict, List, Optional


# PROFILE_SCHEMA_HINT = """Return ONLY valid JSON with this structure (no explanations):
# {
#   "core_preferences": ["keyword", "..."],
#   "avoidances": ["keyword", "..."],
#   "recent_sentiment": "positive|neutral|negative"
# }"""


# class ProfileModuleBase:
#     """
#     Base class for user profile summarizers used by the enhanced agent.
#     """

#     def __init__(self, llm) -> None:
#         if llm is None:
#             raise ValueError("llm must not be None")
#         self.llm = llm

#     def __call__(
#         self,
#         *,
#         user_profile: Optional[Dict[str, Any]],
#         user_reviews: Optional[List[Dict[str, Any]]],
#         interaction_tool=None,
#     ) -> Dict[str, Any]:
#         raise NotImplementedError


# class StructuredProfileModule(ProfileModuleBase):
#     """
#     LLM-backed module that produces a normalized persona JSON blob.
#     """

#     def __call__(
#         self,
#         *,
#         user_profile: Optional[Dict[str, Any]],
#         user_reviews: Optional[List[Dict[str, Any]]],
#         interaction_tool=None,
#     ) -> Dict[str, Any]:
#         if not user_profile and not user_reviews:
#             return {}

#         profile_section = json.dumps(user_profile or {}, ensure_ascii=False, default=str)
#         review_section = self._format_reviews(user_reviews or [])

#         prompt = (
#             "You build concise personas for recommendation systems.\n"
#             "Extract only the key preference fields using the schema below.\n"
#             f"{PROFILE_SCHEMA_HINT}\n\n"
#             f"USER PROFILE JSON:\n{profile_section}\n\n"
#             f"LATEST REVIEWS (already sorted by recency if provided):\n{review_section}\n"
#         )

#         llm_response = self.llm(
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#         )
#         # logging.info("StructuredProfileModule raw LLM response:\n%s", llm_response)
#         extracted_fields = self._safe_json_loads(llm_response)
#         if extracted_fields is None:
#             message = "StructuredProfileModule could not extract persona fields from LLM response."
#             # logging.error("%s Raw response:\n%s", message, llm_response)
#             raise RuntimeError(message)

#         persona = self._build_persona_from_fields(
#             extracted_fields, user_profile, user_reviews or []
#         )
#         return persona

#     @staticmethod
#     def _format_reviews(user_reviews: List[Dict[str, Any]]) -> str:
#         formatted = []
#         for review in user_reviews[:5]:
#             snippet = (review.get("text") or "").strip().replace("\n", " ")
#             if len(snippet) > 200:
#                 snippet = snippet[:197] + "..."
#             formatted.append(
#                 json.dumps(
#                     {
#                         "review_id": review.get("review_id"),
#                         "item_id": review.get("item_id"),
#                         "stars": review.get("stars"),
#                         "text": snippet,
#                     },
#                     ensure_ascii=False,
#                     default=str,
#                 )
#             )
#         return "\n".join(formatted) if formatted else "[]"

#     @staticmethod
#     def _safe_json_loads(raw_output: Any) -> Optional[Dict[str, Any]]:
#         if raw_output is None:
#             return None

#         if StructuredProfileModule._looks_like_persona(raw_output):
#             return raw_output

#         text = StructuredProfileModule._flatten_to_string(raw_output)
#         if not text:
#             return None

#         text = StructuredProfileModule._strip_code_fences(text).strip()
#         if not text:
#             return None

#         candidate = StructuredProfileModule._extract_first_braced_block(text) or text
#         for parser in (json.loads, ast.literal_eval):
#             try:
#                 parsed = parser(candidate)
#             except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
#                 continue
#             if StructuredProfileModule._looks_like_persona(parsed):
#                 return parsed
#         return None

#     @staticmethod
#     def _flatten_to_string(raw_output: Any) -> str:
#         parts: List[str] = []

#         def _collect(obj: Any) -> None:
#             if isinstance(obj, str):
#                 parts.append(obj)
#             elif isinstance(obj, dict):
#                 for value in obj.values():
#                     _collect(value)
#             elif isinstance(obj, (list, tuple, set)):
#                 for item in obj:
#                     _collect(item)
#             elif obj is not None:
#                 parts.append(str(obj))

#         _collect(raw_output)
#         return " ".join(segment for segment in parts if segment)

#     @staticmethod
#     def _strip_code_fences(text: str) -> str:
#         stripped = text.strip()
#         if "```" not in stripped:
#             return stripped

#         segments = [segment.strip() for segment in stripped.split("```") if segment.strip()]
#         if not segments:
#             return stripped

#         candidate = segments[-1]
#         first_line_break = candidate.find("\n")
#         if first_line_break != -1:
#             language_hint = candidate[:first_line_break].strip().lower()
#             if language_hint.isalpha() and len(language_hint) < 10:
#                 return candidate[first_line_break + 1 :].strip()
#         return candidate.strip()

#     @staticmethod
#     def _looks_like_persona(obj: Any) -> bool:
#         if not isinstance(obj, dict):
#             return False
#         required = {"core_preferences", "avoidances", "recent_sentiment"}
#         return required.issubset(obj.keys())

#     @staticmethod
#     def _build_persona_from_fields(
#         fields: Dict[str, Any],
#         user_profile: Optional[Dict[str, Any]],
#         user_reviews: List[Dict[str, Any]],
#     ) -> Dict[str, Any]:
#         preferences = StructuredProfileModule._ensure_str_list(fields.get("core_preferences"))
#         avoidances = StructuredProfileModule._ensure_str_list(fields.get("avoidances"))
#         sentiment = StructuredProfileModule._normalize_sentiment(fields.get("recent_sentiment"))
#         name = (user_profile or {}).get("name") or "The user"
#         summary = StructuredProfileModule._compose_summary(name, sentiment, preferences, avoidances)

#         evidence = StructuredProfileModule._build_evidence(user_reviews)
#         return {
#             "summary": summary,
#             "core_preferences": preferences,
#             "avoidances": avoidances,
#             "recent_sentiment": sentiment,
#             "evidence": evidence,
#         }

#     @staticmethod
#     def _ensure_str_list(value: Any) -> List[str]:
#         if isinstance(value, str):
#             items = [value]
#         elif isinstance(value, (list, tuple, set)):
#             items = [str(v) for v in value if isinstance(v, (str, int, float))]
#         else:
#             items = []
#         return [item.strip() for item in items if item and item.strip()]

#     @staticmethod
#     def _normalize_sentiment(value: Any) -> str:
#         if isinstance(value, str):
#             lowered = value.strip().lower()
#             if lowered in {"positive", "neutral", "negative"}:
#                 return lowered
#         return "neutral"

#     @staticmethod
#     def _compose_summary(name: str, sentiment: str, prefs: List[str], avoidances: List[str]) -> str:
#         parts = [f"{name} shows {sentiment} sentiment."]
#         if prefs:
#             parts.append(f"Enjoys {', '.join(prefs[:3])}.")
#         if avoidances:
#             parts.append(f"Avoids {', '.join(avoidances[:3])}.")
#         return " ".join(parts)

#     @staticmethod
#     def _build_evidence(user_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         evidence = []
#         for review in user_reviews[:5]:
#             snippet = (review.get("text") or "").strip().replace("\n", " ")
#             if len(snippet) > 160:
#                 snippet = snippet[:157] + "..."
#             evidence.append(
#                 {
#                     "review_id": review.get("review_id"),
#                     "item_id": review.get("item_id"),
#                     "snippet": snippet,
#                 }
#             )
#         return evidence

#     @staticmethod
#     def _extract_first_braced_block(raw_output: str) -> Optional[str]:
#         start = None
#         depth = 0
#         for idx, char in enumerate(raw_output):
#             if char == "{":
#                 if depth == 0:
#                     start = idx
#                 depth += 1
#             elif char == "}":
#                 if depth > 0:
#                     depth -= 1
#                     if depth == 0 and start is not None:
#                         return raw_output[start : idx + 1]
#         return None

# __all__ = ["ProfileModuleBase", "StructuredProfileModule"]

