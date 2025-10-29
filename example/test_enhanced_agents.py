"""
Quick Start Script for Testing Enhanced Agents

This script provides a simple way to test the enhanced agents with your data.
Modify the configuration section below to match your setup.
"""

import json
import logging
import argparse
from pathlib import Path

# Import the agents
from EnhancedSimulationAgent import EnhancedSimulationAgent
from EnhancedRecommendationAgent import EnhancedRecommendationAgent

# Import the simulator and LLM
from websocietysimulator import Simulator
from websocietysimulator.llm import InfinigenceLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_simulation_agent(args):
    """
    Test the Enhanced Simulation Agent (Track 1)
    """
    logger.info("="*60)
    logger.info("Testing Enhanced Simulation Agent (Track 1)")
    logger.info("="*60)
    
    # Initialize Simulator
    logger.info(f"Loading data from: {args.data_dir}")
    simulator = Simulator(
        data_dir=args.data_dir,
        device=args.device,
        cache=args.cache
    )
    
    # Load tasks and ground truth
    task_dir = f"./track1/{args.dataset}/tasks"
    groundtruth_dir = f"./track1/{args.dataset}/groundtruth"
    
    logger.info(f"Loading tasks from: {task_dir}")
    logger.info(f"Loading groundtruth from: {groundtruth_dir}")
    
    simulator.set_task_and_groundtruth(
        task_dir=task_dir,
        groundtruth_dir=groundtruth_dir
    )
    
    # Set agent
    simulator.set_agent(EnhancedSimulationAgent)
    logger.info("Enhanced Simulation Agent loaded successfully")
    
    # Set LLM
    simulator.set_llm(InfinigenceLLM(api_key=args.api_key))
    logger.info("LLM configured")
    
    # Run simulation
    logger.info(f"Running simulation with {args.num_tasks or 'ALL'} tasks...")
    logger.info(f"Threading: {args.enable_threading}, Workers: {args.max_workers}")
    
    outputs = simulator.run_simulation(
        number_of_tasks=args.num_tasks,
        enable_threading=args.enable_threading,
        max_workers=args.max_workers
    )
    
    logger.info(f"Simulation complete! Processed {len(outputs)} tasks")
    
    # Evaluate
    logger.info("Evaluating results...")
    evaluation_results = simulator.evaluate()
    
    # Display results
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    for metric, value in evaluation_results.items():
        logger.info(f"{metric}: {value}")
    logger.info("="*60)
    
    # Save results
    output_file = f'./evaluation_results_enhanced_track1_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    logger.info(f"Results saved to: {output_file}")
    
    return evaluation_results


def test_recommendation_agent(args):
    """
    Test the Enhanced Recommendation Agent (Track 2)
    """
    logger.info("="*60)
    logger.info("Testing Enhanced Recommendation Agent (Track 2)")
    logger.info("="*60)
    
    # Initialize Simulator
    logger.info(f"Loading data from: {args.data_dir}")
    simulator = Simulator(
        data_dir=args.data_dir,
        device=args.device,
        cache=args.cache
    )
    
    # Load tasks and ground truth
    task_dir = f"./track2/{args.dataset}/tasks"
    groundtruth_dir = f"./track2/{args.dataset}/groundtruth"
    
    logger.info(f"Loading tasks from: {task_dir}")
    logger.info(f"Loading groundtruth from: {groundtruth_dir}")
    
    simulator.set_task_and_groundtruth(
        task_dir=task_dir,
        groundtruth_dir=groundtruth_dir
    )
    
    # Set agent
    simulator.set_agent(EnhancedRecommendationAgent)
    logger.info("Enhanced Recommendation Agent loaded successfully")
    
    # Set LLM
    simulator.set_llm(InfinigenceLLM(api_key=args.api_key))
    logger.info("LLM configured")
    
    # Run simulation
    logger.info(f"Running simulation with {args.num_tasks or 'ALL'} tasks...")
    logger.info(f"Threading: {args.enable_threading}, Workers: {args.max_workers}")
    
    outputs = simulator.run_simulation(
        number_of_tasks=args.num_tasks,
        enable_threading=args.enable_threading,
        max_workers=args.max_workers
    )
    
    logger.info(f"Simulation complete! Processed {len(outputs)} tasks")
    
    # Evaluate
    logger.info("Evaluating results...")
    evaluation_results = simulator.evaluate()
    
    # Display results
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    for metric, value in evaluation_results.items():
        logger.info(f"{metric}: {value}")
    logger.info("="*60)
    
    # Save results
    output_file = f'./evaluation_results_enhanced_track2_{args.dataset}.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    logger.info(f"Results saved to: {output_file}")
    
    return evaluation_results


def main():
    """
    Main function with argument parsing
    """
    parser = argparse.ArgumentParser(
        description='Test Enhanced Agents for AgentSociety Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Track 1 (Simulation) with 5 tasks on Yelp dataset
  python test_enhanced_agents.py --track 1 --dataset yelp --num-tasks 5 --api-key YOUR_KEY

  # Test Track 2 (Recommendation) with all tasks on Amazon dataset
  python test_enhanced_agents.py --track 2 --dataset amazon --api-key YOUR_KEY

  # Test with threading enabled
  python test_enhanced_agents.py --track 1 --dataset yelp --enable-threading --max-workers 10 --api-key YOUR_KEY
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--track',
        type=int,
        choices=[1, 2],
        required=True,
        help='Track number: 1 for Simulation, 2 for Recommendation'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['yelp', 'amazon', 'goodreads'],
        required=True,
        help='Dataset to use: yelp, amazon, or goodreads'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='API key for the LLM service (e.g., Infinigence or OpenAI)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data_processed',
        help='Path to processed data directory (default: ./data_processed)'
    )
    
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=None,
        help='Number of tasks to run (default: None, runs all tasks)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu'],
        help='Device to use (default: auto)'
    )
    
    parser.add_argument(
        '--cache',
        action='store_true',
        default=True,
        help='Use cache for interaction tool (default: True)'
    )
    
    parser.add_argument(
        '--no-cache',
        dest='cache',
        action='store_false',
        help='Disable cache (loads all data into memory)'
    )
    
    parser.add_argument(
        '--enable-threading',
        action='store_true',
        default=False,
        help='Enable multi-threading for faster execution'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of worker threads (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Please run data_process.py first to prepare the data")
        return
    
    # Run the appropriate agent
    try:
        if args.track == 1:
            results = test_simulation_agent(args)
        else:
            results = test_recommendation_agent(args)
        
        logger.info("Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

