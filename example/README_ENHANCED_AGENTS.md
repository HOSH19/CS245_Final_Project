# Enhanced Agents for AgentSociety Challenge

Complete implementation of modular agents using the reasoning, planning, and memory modules from the websocietysimulator framework.

## 🚀 Quick Start

### Using Google GCP Credits (Recommended - You Have $50!)

```bash
# 1. Get your Google API key from: https://aistudio.google.com/app/apikey
# 2. Install Google's package
pip install google-generativeai

# 3. Set your API key
export GOOGLE_API_KEY="your-google-api-key"

# 4. Test Track 1
cd example
python -c "
from GoogleGeminiLLM import GoogleGeminiLLM
from websocietysimulator import Simulator
from EnhancedSimulationAgent import EnhancedSimulationAgent

llm = GoogleGeminiLLM(model='gemini-1.5-flash')
simulator = Simulator(data_dir='../data_processed', cache=True)
simulator.set_task_and_groundtruth(
    task_dir='./track1/yelp/tasks',
    groundtruth_dir='./track1/yelp/groundtruth'
)
simulator.set_agent(EnhancedSimulationAgent)
simulator.set_llm(llm)
outputs = simulator.run_simulation(number_of_tasks=5)
print(simulator.evaluate())
"
```

**See `SETUP_GOOGLE_GCP.md` for detailed instructions!**

### Using Other API Keys

If you prefer OpenAI or other providers:

```bash
cd example

# With OpenAI
python test_enhanced_agents.py --track 1 --dataset yelp --num-tasks 5 --api-key YOUR_OPENAI_KEY

# With Infinigence
python test_enhanced_agents.py --track 2 --dataset yelp --num-tasks 5 --api-key YOUR_INFINIGENCE_KEY
```

Results will be saved to `evaluation_results_enhanced_track{1,2}_{dataset}.json`

---

## 🤖 LLM Options

The framework supports multiple LLM providers:

| Provider | Cost | Setup | File |
|----------|------|-------|------|
| **Google Gemini** ✅ | $0.075/1M tokens | [Guide](SETUP_GOOGLE_GCP.md) | `GoogleGeminiLLM.py` |
| **OpenAI GPT** | $0.50/1M tokens | Get key at [platform.openai.com](https://platform.openai.com) | Built-in `OpenAILLM` |
| **Infinigence** | Variable | Get key at [cloud.infini-ai.com](https://cloud.infini-ai.com) | Built-in `InfinigenceLLM` |

**For your $50 GCP credits**: Use Google Gemini! See `SETUP_GOOGLE_GCP.md` for complete setup.

---

## 📦 What Was Implemented

### 1. Enhanced Agents (2 files)

#### **EnhancedSimulationAgent.py** - Track 1 (User Behavior Simulation)
- **Planning Module** (`PlanningIO`): Creates structured plans for information gathering
- **Reasoning Module** (`ReasoningCOT`): Uses Chain-of-Thought for step-by-step review generation
- **Memory Module** (`MemoryDILU`): Vector-based storage and retrieval of relevant reviews
- **Features**: Token management, error handling, robust parsing, detailed logging

#### **EnhancedRecommendationAgent.py** - Track 2 (Recommendation)
- **Planning Module** (`PlanningIO`): Systematically analyzes user preferences
- **Reasoning Module** (`ReasoningStepBack`): Extracts high-level principles before ranking
- **Memory Module** (`MemoryGenerative`): Importance-scored retrieval of review patterns
- **Features**: Advanced token management with tiktoken, preference analysis, validation

### 2. Testing Tool

#### **test_enhanced_agents.py** - CLI Testing Script
Easy-to-use command-line tool for testing both agents with any dataset.

---

## 🎯 Key Features

### What Makes These "Enhanced"?

| Feature | Baseline | Enhanced |
|---------|----------|----------|
| **Architecture** | Monolithic | Modular (swappable components) |
| **Planning** | Hardcoded steps | Intelligent planning modules |
| **Reasoning** | Simple prompts | CoT/StepBack structured reasoning |
| **Memory** | None/basic | Vector-based similarity search |
| **Token Management** | Basic slicing | Intelligent truncation with tiktoken |
| **Error Handling** | Minimal | Comprehensive with fallbacks |
| **Customization** | Edit code | Swap modules (2 lines) |

### Modular Architecture

Both agents use swappable modules:

**Planning Modules** (6 available):
- `PlanningIO` ✅ - Basic structured planning (used in enhanced agents)
- `PlanningVoyager` - Goal-oriented planning
- `PlanningHUGGINGGPT` - Dependency-aware planning
- `PlanningOPENAGI` - Concise todo lists
- `PlanningDEPS` - Multi-hop sub-goals
- `PlanningTD` - Temporal dependencies

**Reasoning Modules** (7 available):
- `ReasoningIO` - Direct reasoning
- `ReasoningCOT` ✅ - Chain-of-Thought (used in Track 1)
- `ReasoningCOTSC` - CoT with Self-Consistency
- `ReasoningTOT` - Tree-of-Thought
- `ReasoningDILU` - Example-guided reasoning
- `ReasoningSelfRefine` - Iterative refinement
- `ReasoningStepBack` ✅ - Principle extraction (used in Track 2)

**Memory Modules** (4 available):
- `MemoryDILU` ✅ - Direct similarity retrieval (used in Track 1)
- `MemoryGenerative` ✅ - Importance scoring (used in Track 2)
- `MemoryTP` - Experience-based planning
- `MemoryVoyager` - Summarized trajectories

---

## 📊 How It Works

### Track 1: User Behavior Simulation

```
1. Planning Phase
   └─> Create information gathering plan

2. Execution Phase
   ├─> Get user profile
   ├─> Get business/item info
   ├─> Get other users' reviews → Store in Memory
   └─> Get user's review history

3. Memory Phase
   └─> Query for most relevant context

4. Reasoning Phase (Chain-of-Thought)
   ├─> Analyze user profile and patterns
   ├─> Consider business characteristics
   ├─> Review what others said
   ├─> Match user's writing style
   └─> Generate authentic review

5. Output
   └─> {stars: float, review: string}
```

### Track 2: Recommendation

```
1. Planning Phase
   └─> Create recommendation strategy

2. Execution Phase
   ├─> Get user profile
   ├─> Get user review history → Store in Memory
   └─> Get all candidate items

3. Analysis Phase
   ├─> Calculate rating statistics
   ├─> Extract preference patterns
   └─> Identify likes/dislikes

4. Reasoning Phase (Step-Back)
   ├─> Extract high-level principles
   │   (What categories/features user prefers)
   └─> Apply principles to rank items

5. Validation Phase
   ├─> Remove duplicates
   └─> Ensure all candidates included

6. Output
   └─> [item_id1, item_id2, ..., item_id20]
```

---

## 🛠️ Usage Examples

### Command Line (Easiest)

```bash
# Quick test with 5 tasks
python test_enhanced_agents.py --track 1 --dataset yelp --num-tasks 5 --api-key YOUR_KEY

# Test different dataset
python test_enhanced_agents.py --track 2 --dataset amazon --num-tasks 10 --api-key YOUR_KEY

# Full evaluation with threading (faster)
python test_enhanced_agents.py --track 1 --dataset yelp --enable-threading --max-workers 10 --api-key YOUR_KEY

# All available options
python test_enhanced_agents.py --help
```

### Python Script

```python
from websocietysimulator import Simulator
from EnhancedSimulationAgent import EnhancedSimulationAgent
from websocietysimulator.llm import InfinigenceLLM

# Initialize
simulator = Simulator(data_dir="../data_processed", cache=True)
simulator.set_task_and_groundtruth(
    task_dir="./track1/yelp/tasks",
    groundtruth_dir="./track1/yelp/groundtruth"
)

# Set agent and LLM
simulator.set_agent(EnhancedSimulationAgent)
simulator.set_llm(InfinigenceLLM(api_key="YOUR_API_KEY"))

# Run and evaluate
outputs = simulator.run_simulation(number_of_tasks=10)
results = simulator.evaluate()
print(results)
```

---

## 🔧 Customization

### Easy: Swap Modules (2 lines)

```python
# Change reasoning strategy
from websocietysimulator.agent.modules.reasoning_modules import ReasoningTOT

class MyAgent(SimulationAgent):
    def __init__(self, llm):
        super().__init__(llm)
        # Change from ReasoningCOT to ReasoningTOT
        self.reasoning = ReasoningTOT(profile_type_prompt='', memory=None, llm=llm)
```

### Medium: Customize Prompts

```python
class MyCustomPlanning(PlanningIO):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        # Write your own planning prompt
        return f"""
        Custom planning prompt:
        {task_description}
        
        Create a detailed plan...
        """
```

### Advanced: Create New Modules

```python
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase

class MyCustomReasoning(ReasoningBase):
    def __call__(self, task_description: str, feedback: str = ''):
        # Implement your custom reasoning logic
        prompt = f"Custom reasoning: {task_description}"
        result = self.llm(messages=[{"role": "user", "content": prompt}])
        return result
```

---

## 📈 Understanding Results

### Track 1 Metrics
- **RMSE**: Root Mean Square Error for rating prediction (lower is better)
- **Sentiment Alignment**: How well review sentiment matches ground truth (higher is better)

### Track 2 Metrics
- **HR@1**: Hit Rate at position 1 - Did we get the top item correct?
- **HR@3**: Hit Rate at top 3 - Is the correct item in top 3?
- **HR@5**: Hit Rate at top 5 - Is the correct item in top 5?

---

## 🐛 Troubleshooting

### "Module not found"
```bash
cd .. # Go to root
poetry install && poetry shell
# OR
pip install -r requirements.txt
```

### "Data directory not found"
```bash
cd .. # Go to root
python data_process.py --input <path_to_raw_data> --output ./data_processed
```

### Out of Memory
```bash
# Use cache mode (default) and fewer tasks
python test_enhanced_agents.py --track 1 --dataset yelp --num-tasks 10 --cache --api-key YOUR_KEY
```

### API Rate Limiting
```bash
# Reduce workers or disable threading
python test_enhanced_agents.py --track 1 --dataset yelp --max-workers 3 --api-key YOUR_KEY
```

### Slow Execution
```bash
# Enable threading for parallel processing
python test_enhanced_agents.py --track 1 --dataset yelp --enable-threading --max-workers 10 --api-key YOUR_KEY
```

---

## 🏆 Competition Submission

### Preparation Steps

1. **Test locally** with all datasets
   ```bash
   python test_enhanced_agents.py --track 1 --dataset yelp --num-tasks 50 --api-key YOUR_KEY
   ```

2. **Create submission file**
   ```bash
   cp EnhancedSimulationAgent.py YourTeamName_Track1.py
   ```

3. **Clean up the file**
   - ❌ Remove the `if __name__ == "__main__":` block
   - ❌ Remove test code and API keys
   - ❌ Remove excessive comments
   - ✅ Keep all imports and class definitions
   - ✅ Keep workflow implementation

4. **Final structure**
   ```python
   # YourTeamName_Track1.py
   
   from websocietysimulator.agent import SimulationAgent
   from websocietysimulator.llm import LLMBase
   from websocietysimulator.agent.modules.planning_modules import PlanningIO
   from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOT
   from websocietysimulator.agent.modules.memory_modules import MemoryDILU
   
   # Your custom modules (if any)
   class CustomPlanning(PlanningIO):
       # ...
   
   # Your agent
   class YourTeamNameAgent(SimulationAgent):
       def __init__(self, llm: LLMBase):
           super().__init__(llm=llm)
           # Initialize modules
           
       def workflow(self):
           # Your implementation
           return result
   
   # NO __main__ block for submission!
   ```

5. **Submit** the single `.py` file to the competition website

---

## 💡 Pro Tips

1. **Start small**: Always test with `--num-tasks 5` first
2. **Use threading**: 10x faster with `--enable-threading --max-workers 10`
3. **Track costs**: Monitor API usage, especially with threading enabled
4. **Experiment**: Try all module combinations to find what works best
5. **Read the code**: Agent implementations are well-documented
6. **Check logs**: Logging helps identify issues quickly
7. **Version control**: Use git to track your experiments

---

## 📚 Architecture Details

### Module Interaction

```
Task Input
    ↓
Planning Module
    ↓
Information Gathering (via Interaction Tool)
    ↓
Memory Module (Store & Retrieve)
    ↓
Data Summarization
    ↓
Reasoning Module (CoT or StepBack)
    ↓
Parse & Validate
    ↓
Output Result
```

### Key Components

**Interaction Tool** (provided by framework):
- `get_user(user_id)` - Retrieve user profile
- `get_item(item_id)` - Retrieve item/business details
- `get_reviews(user_id/item_id)` - Retrieve reviews

**Planning Module**:
- Input: Task description, task type, feedback
- Output: List of subtasks with instructions
- Purpose: Structure the information gathering process

**Reasoning Module**:
- Input: Task description with gathered context
- Output: Generated result (review or ranking)
- Purpose: Generate intelligent outputs using structured reasoning

**Memory Module**:
- Input: Text to store or query
- Storage: Vector database (Chroma) with embeddings
- Purpose: Store and retrieve relevant information

---

## 🎓 Next Steps

### Immediate (5 minutes)
```bash
cd example
python test_enhanced_agents.py --track 1 --dataset yelp --num-tasks 5 --api-key YOUR_KEY
```

### Today (1 hour)
- Test both tracks
- Try different datasets (yelp, amazon, goodreads)
- Read the agent source code

### This Week
- Experiment with different module combinations
- Customize prompts for better performance
- Run full evaluation on all tasks

### Before Submission
- Fine-tune your best configuration
- Test on all datasets
- Clean up and prepare submission file
- Submit to competition!

---

## 📞 Resources

- **Competition Homepage**: https://tsinghua-fib-lab.github.io/AgentSocietyChallenge
- **Competition Platform**: https://www.codabench.org/competitions/4574/
- **Paper**: https://arxiv.org/abs/2502.18754
- **Main README**: ../README.md
- **Tutorials**: ../tutorials/

---

## 🎉 You're Ready!

You now have:
- ✅ Two production-ready modular agents
- ✅ 17+ module options to experiment with
- ✅ Easy testing infrastructure
- ✅ Comprehensive error handling and logging
- ✅ Token management for cost efficiency
- ✅ Competition-ready submission format

**Start testing now:**
```bash
python test_enhanced_agents.py --track 1 --dataset yelp --num-tasks 5 --api-key YOUR_API_KEY
```

Good luck with the AgentSociety Challenge! 🚀

