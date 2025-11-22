import re
import ast

class SchemaFitterBase:
    def __init__(self, llm):
        """
        Base class for schema-fitting summarizers.

        Args:
            llm: an LLM instance with a __call__(messages=[...]) interface.
        """
        self.llm = llm
        self.output = None

    def create_prompt(self, schema, items):
        """
        Subclasses will define how the system prompt is constructed.
        """
        raise NotImplementedError

    def __call__(self, schema, items):
        """
        Execute the summarization / schema-fitting call.
        """
        prompt = self.create_prompt(schema, items)

        messages = [{"role": "user", "content": prompt}]
        raw_output = self.llm(messages=messages, temperature=0.1)

        # Extract JSON-like dicts from output
        dict_strings = re.findall(r"\{[^{}]*\}", raw_output)
        dicts = []

        for ds in dict_strings:
            try:
                dicts.append(ast.literal_eval(ds))
            except Exception:
                # Ignore bad parses
                pass

        self.output = dicts
        return dicts
    
class SchemaFitterIO(SchemaFitterBase):
    def create_prompt(self, schema, items):
        """
        items can be:
          - a single dict: {"user_id": ..., "user": {...}, "reviews": [...]}
          - a list of such dicts
        """

        prompt = f"""
You are a data summarizer and schema-fitter.

You will receive:
- A SCHEMA describing a structured "user preference profile".
- A list of USER PACKETS. Each packet has:
    - "user_id": the user identifier
    - "user": raw user fields
    - "reviews": a list of that user's reviews across items

For EACH user packet, produce ONE JSON object that matches the SCHEMA.
Include the same "user_id" field in the output so we can align profiles.

SCHEMA:
{schema}

USER PACKETS (list):
{items}

Output:
- A JSON ARRAY.
- Each element is one profile object matching the schema.
- Do NOT output any explanation text, only the JSON array.
"""
        return prompt