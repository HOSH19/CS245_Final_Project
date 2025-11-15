"""
Enhanced Recommendation Agent composed from smaller, easier-to-maintain pieces.
"""

import json
import logging

from websocietysimulator import Simulator
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.planning_modules import PlanningIO
from websocietysimulator.agent.modules.reasoning_modules import ReasoningStepBack
from websocietysimulator.llm import InfinigenceLLM

from enhanced_agent.base_agent import EnhancedRecommendationAgentBase
from websocietysimulator.agent.modules.profile_module import StructuredProfileModule
from enhanced_agent.workflow_mixins import EnhancedWorkflowMixin

logging.basicConfig(level=logging.INFO)


class EnhancedRecommendationAgent(EnhancedWorkflowMixin, EnhancedRecommendationAgentBase):
    """
    Final concrete agent that bundles the base functionality with the library of
    workflow combinations.
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


if __name__ == "__main__":
    logging.info("Initializing simulator...")
    task_set = "yelp"
    data_dir = "./data_processed"

    simulator = Simulator(data_dir=data_dir, device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./track2/{task_set}/tasks",
        groundtruth_dir=f"./track2/{task_set}/groundtruth",
    )
    simulator.set_agent(EnhancedRecommendationAgent)
    simulator.set_llm(InfinigenceLLM(api_key="your_api_key_here"))

    logging.info("Starting recommendation simulation...")
    simulator.run_simulation(number_of_tasks=10, enable_threading=True, max_workers=5)
    logging.info("Evaluating results...")
    evaluation_results = simulator.evaluate()

    output_file = f"./evaluation_results_enhanced_track2_{task_set}.json"
    with open(output_file, "w", encoding="utf-8") as file_handle:
        json.dump(evaluation_results, file_handle, indent=4)

    logging.info("Evaluation complete! Results saved to %s", output_file)
    logging.info("Results: %s", evaluation_results)

