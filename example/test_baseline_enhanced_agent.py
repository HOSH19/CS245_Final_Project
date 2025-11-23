"""
Test workflow function for testing different recommendation workflows.

Usage:
    python test_baseline_enhanced_agent.py --workflow openagi --num-tasks 5
    python test_baseline_enhanced_agent.py --workflow default --task-set goodreads --num-tasks 10
"""

import argparse
import logging
import os
import time

from websocietysimulator import Simulator
from GoogleGeminiLLM import GoogleGeminiLLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def create_workflow_agent(workflow_name, llm):
    """
    Create an agent class that uses a specific workflow.
    
    Args:
        workflow_name: Name of the workflow to use
        llm: LLM instance
    """
    from EnhancedRecommendationAgent import EnhancedRecommendationAgent
    
    if workflow_name == 'default':
        return EnhancedRecommendationAgent
    
    class WorkflowAgent(EnhancedRecommendationAgent):
        def workflow(self):
            # Map workflow names to methods
            workflow_map = {
                # Original workflows
                'voyager': self.workflow_with_voyager_planning,
                'self_refine': self.workflow_with_self_refine,
                'cot_sc': self.workflow_with_cot_sc,
                'voyager_memory': self.workflow_with_voyager_memory,
                'openagi': self.workflow_with_openagi_planning,
                'hybrid': self.workflow_hybrid_advanced,
                # New workflows exploring unused modules
                'tot': self.workflow_with_tot_reasoning,
                'td': self.workflow_with_td_planning,
                'deps': self.workflow_with_deps_planning,
                'all_voyager': self.workflow_all_voyager,
                'dilu_memory': self.workflow_with_dilu_memory,
                'simple': self.workflow_simple_efficient,
                'tot_memory': self.workflow_tot_with_memory,
                'deps_refine': self.workflow_deps_self_refine
            }
            
            if workflow_name in workflow_map:
                return workflow_map[workflow_name]()
            else:
                raise ValueError(f"Unknown workflow: {workflow_name}")
    
    return WorkflowAgent


def test_workflow(workflow_name, dataset='goodreads', num_tasks=10, llm_model='gemini-2.0-flash', data_dir=None):
    """
    Test a specific workflow and return accuracy metrics.
    
    Args:
        workflow_name: Name of the workflow to test
        dataset: Dataset to use ('yelp', 'amazon', or 'goodreads')
        num_tasks: Number of tasks to test
        llm_model: LLM model to use
        data_dir: Path to processed data directory
    
    Returns:
        dict: Contains metrics and timing information
    """
    print(f"\n{'='*80}")
    print(f"Testing: {workflow_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Initialize LLM
        llm = GoogleGeminiLLM(model=llm_model)
        
        # Initialize Simulator
        if data_dir is None:
            data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data_processed"))
        simulator = Simulator(data_dir=data_dir, cache=True)
        simulator.set_task_and_groundtruth(
            task_dir=f'./track2/{dataset}/tasks',
            groundtruth_dir=f'./track2/{dataset}/groundtruth'
        )
        
        # Set the agent with specific workflow
        WorkflowAgent = create_workflow_agent(workflow_name, llm)
        simulator.set_agent(WorkflowAgent)
        simulator.set_llm(llm)
        
        # Run simulation
        print(f"Running {num_tasks} tasks...")
        start_time = time.time()
        try:
            simulator.run_simulation(number_of_tasks=num_tasks)
            execution_time = time.time() - start_time

            # Evaluate
            print("Evaluating results...")
            metrics = simulator.evaluate()
        except Exception as exc:
            execution_time = time.time() - start_time
            print(f"\n✗ Error during workflow '{workflow_name}': {exc}")
            metrics = {"error": str(exc)}
            return {
                "workflow": workflow_name,
                "metrics": metrics,
                "execution_time": execution_time,
                "avg_time_per_task": execution_time / max(num_tasks, 1),
                "num_tasks": num_tasks,
                "dataset": dataset,
                "success": False,
            }
        
        # Add timing info
        result = {
            'workflow': workflow_name,
            'metrics': metrics,
            'execution_time': execution_time,
            'avg_time_per_task': execution_time / num_tasks,
            'num_tasks': num_tasks,
            'dataset': dataset,
            'success': True
        }
        
        # Print results
        print(f"\n✓ Completed in {execution_time:.1f}s ({execution_time/num_tasks:.1f}s per task)")
        print(f"\nMetrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'workflow': workflow_name,
            'error': str(e),
            'success': False
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Test workflow function for recommendation workflows")
    parser.add_argument(
        "--workflow",
        required=True,
        choices=['default', 'voyager', 'self_refine', 'cot_sc', 'voyager_memory', 'openagi', 'hybrid',
                 'tot', 'td', 'deps', 'all_voyager', 'dilu_memory', 'simple', 'tot_memory', 'deps_refine'],
        help="Workflow to test",
    )
    parser.add_argument("--task-set", default="goodreads", choices=["yelp", "amazon", "goodreads"],
                        help="Dataset to use (default: goodreads)")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to test (default: 10)")
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

    result = test_workflow(
        workflow_name=args.workflow,
        dataset=args.task_set,
        num_tasks=args.num_tasks,
        llm_model=args.model,
        data_dir=args.data_dir
    )
    
    if result.get('success'):
        logging.info("Workflow test complete.")
    else:
        logging.error("Workflow test failed.")


if __name__ == "__main__":
    main()

