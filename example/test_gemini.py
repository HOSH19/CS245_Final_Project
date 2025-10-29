from GoogleGeminiLLM import GoogleGeminiLLM
from websocietysimulator import Simulator
from EnhancedSimulationAgent import EnhancedSimulationAgent

llm = GoogleGeminiLLM(model='gemini-1.5-flash')  # Reads from .env automatically
simulator = Simulator(data_dir='../data_processed', cache=True)
simulator.set_task_and_groundtruth(
    task_dir='./track1/yelp/tasks',
    groundtruth_dir='./track1/yelp/groundtruth'
)
simulator.set_agent(EnhancedSimulationAgent)
simulator.set_llm(llm)
outputs = simulator.run_simulation(number_of_tasks=5)
print(simulator.evaluate())