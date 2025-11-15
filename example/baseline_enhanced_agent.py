"""
Diagnostic baseline agent to sanity-check LLM connectivity and module outputs.

Run it to observe:
1. The generated plan from the selected planning module
2. Gathered context (user profile/reviews/candidate items)
3. Memory storage/retrieval behavior
4. Raw reasoning output and the final ranked list
"""

import argparse
import json
import logging
import os

from websocietysimulator import Simulator
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.planning_modules import PlanningIO
from websocietysimulator.agent.modules.reasoning_modules import ReasoningStepBack
from GoogleGeminiLLM import GoogleGeminiLLM

from enhanced_agent.base_agent import EnhancedRecommendationAgentBase
from enhanced_agent.utils import parse_recommendation_result, validate_recommendations

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class BaselineEnhancedAgent(EnhancedRecommendationAgentBase):
    """
    Minimal diagnostic agent that logs intermediate outputs from each module.
    """

    def __init__(self, llm):
        super().__init__(
            llm=llm,
            planning_module=PlanningIO(llm),
            memory_module=MemoryGenerative(llm),
            reasoning_module=ReasoningStepBack(
                profile_type_prompt="You are an intelligent recommendation system.",
                memory=None,
                llm=llm,
            ),
        )

    def workflow(self):
        plan = self._generate_plan(self.planning)
        logging.info("Generated plan (%d steps):\n%s", len(plan), json.dumps(plan, indent=2))

        context = self._gather_context(plan)
        logging.info(
            "Context gathered | user_reviews=%d | candidate_items=%d",
            len(context["user_reviews"]),
            len(context["candidate_items"]),
        )

        enriched_context = self._integrate_memory(context, self.memory)
        if enriched_context.get("memory_context"):
            logging.info("Memory context snippet:\n%s", enriched_context["memory_context"])
        else:
            logging.info("Memory context empty (no retrieval available).")

        payload = {
            "user_profile": enriched_context.get("user_profile"),
            "user_reviews": enriched_context.get("user_reviews"),
            "candidate_items": enriched_context.get("candidate_items"),
            "candidate_list": self.task["candidate_list"],
            "plan": plan,
        }
        if enriched_context.get("memory_context"):
            payload["memory_context"] = enriched_context["memory_context"]

        task_description = (
            "You are a recommendation system. Use the JSON context below to rank "
            "candidate items for the specified user. Return ONLY a Python list of "
            "item IDs drawn from candidate_list.\n"
            f"CONTEXT:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        logging.info("Invoking reasoning module...")
        reasoning_output = self.reasoning(task_description, user_id=self.task["user_id"])
        logging.info("Raw reasoning output:\n%s", reasoning_output)

        ranked_list = parse_recommendation_result(reasoning_output)
        validated = validate_recommendations(ranked_list, self.task["candidate_list"])
        logging.info("Validated recommendation list:\n%s", validated)
        return validated


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline diagnostic agent runner")
    parser.add_argument("--task-set", default="yelp", choices=["yelp", "amazon", "goodreads"])
    parser.add_argument("--num-tasks", type=int, default=1, help="Number of tasks to run (default: 1)")
    default_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data_processed"))
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help=f"Path to processed data directory (default: {default_data_dir})",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model identifier for InfinigenceLLM (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--api-key",
        default="your_api_key_here",
        help="API key for the selected LLM backend",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    llm = GoogleGeminiLLM(model=args.model)

    simulator = Simulator(data_dir=args.data_dir, device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./track2/{args.task_set}/tasks",
        groundtruth_dir=f"./track2/{args.task_set}/groundtruth",
    )
    simulator.set_agent(BaselineEnhancedAgent)
    simulator.set_llm(llm)

    logging.info("Running %d diagnostic task(s)...", args.num_tasks)
    simulator.run_simulation(number_of_tasks=args.num_tasks)
    logging.info("Diagnostic run complete.")


if __name__ == "__main__":
    main()

