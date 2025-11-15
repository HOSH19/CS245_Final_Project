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
from websocietysimulator.agent.modules.profile_module import StructuredProfileModule
from enhanced_agent.utils import parse_recommendation_result, validate_recommendations

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class GenericWorkflowBaselineAgent(EnhancedRecommendationAgentBase):
    """
    Companion agent that exercises EnhancedRecommendationAgentBase.workflow.
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
            profile_module=StructuredProfileModule(llm),
        )


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
    simulator.set_llm(llm)

    simulator.set_agent(GenericWorkflowBaselineAgent)
    logging.info("Running %d task(s) via EnhancedRecommendationAgentBase.workflow...", args.num_tasks)
    simulator.run_simulation(number_of_tasks=args.num_tasks)
    logging.info("Diagnostic run complete.")


if __name__ == "__main__":
    main()

