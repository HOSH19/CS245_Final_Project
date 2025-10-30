from GoogleGeminiLLM import GoogleGeminiLLM
from websocietysimulator import Simulator
from EnhancedRecommendationAgent import EnhancedRecommendationAgent  # Different agent

llm = GoogleGeminiLLM(model='gemini-2.0-flash')
simulator = Simulator(data_dir='../data_processed', cache=True)
simulator.set_task_and_groundtruth(
    task_dir='./track2/yelp/tasks',    # Track 2 is for RECOMMENDATION
    groundtruth_dir='./track2/yelp/groundtruth'
)
simulator.set_agent(EnhancedRecommendationAgent)  # Different agent
simulator.set_llm(llm)
outputs = simulator.run_simulation(number_of_tasks=5)  # Same method for both tracks
print(simulator.evaluate())