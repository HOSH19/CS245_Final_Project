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
from websocietysimulator.agent.modules.info_orchestrator_module import InfoOrchestrator
from websocietysimulator.agent.modules.schemafitter_module import SchemaFitterIO

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class GenericWorkflowBaselineAgent(EnhancedRecommendationAgentBase):
    """
    Companion agent that exercises EnhancedRecommendationAgentBase.workflow.
    Uses InfoOrchestrator for profile generation.
    """

    def __init__(self, llm):
        """
        Initialize the baseline agent.
        
        Args:
            llm: LLM instance
        """
        planning = PlanningIO(llm)
        memory = MemoryGenerative(llm)
        reasoning = ReasoningStepBack(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=llm,
        )
        
        # Initialize InfoOrchestrator
        # SchemaFitterIO will be initialized with interaction_tool later
        info_orchestrator = InfoOrchestrator(
            memory=memory,
            llm=llm,
            schema_fitter=None,  # Will be set when interaction_tool is available
            interaction_tool=None  # Will be set when interaction_tool is available
        )
        
        super().__init__(
            llm=llm,
            planning_module=planning,
            memory_module=memory,
            reasoning_module=reasoning,
            info_orchestrator=info_orchestrator,
        )
        
        # Store for later initialization
        self._schema_fitter_llm = llm
    
    def insert_task(self, task):
        """
        Override insert_task to initialize InfoOrchestrator with interaction_tool.
        """
        super().insert_task(task)
        
        # Initialize InfoOrchestrator's schema_fitter and interaction_tool
        if self.info_orchestrator and self.interaction_tool:
            if self.info_orchestrator.schema_fitter is None:
                schema_fitter = SchemaFitterIO(self._schema_fitter_llm, self.interaction_tool)
                self.info_orchestrator.schema_fitter = schema_fitter
                self.info_orchestrator.interaction_tool = self.interaction_tool


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline diagnostic agent runner")
    parser.add_argument("--task-set", default="goodreads", choices=["yelp", "amazon", "goodreads"])
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
        help="LLM model identifier (default: gemini-2.0-flash). "
             "Note: API key must be set via GOOGLE_API_KEY environment variable.",
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
    logging.info("Running %d task(s) via EnhancedRecommendationAgentBase.workflow with InfoOrchestrator...", 
                 args.num_tasks)
    simulator.run_simulation(number_of_tasks=args.num_tasks)
    logging.info("Diagnostic run complete.")


if __name__ == "__main__":
    main()

