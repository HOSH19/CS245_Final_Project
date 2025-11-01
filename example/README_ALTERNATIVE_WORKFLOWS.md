# Alternative Workflows

## 1. Workflow Components

Each workflow uses different combinations of Planning, Reasoning, and Memory modules:

### Recommendation Agent (Track 2)

| Workflow Name | Planning Module | Reasoning Module | Memory Module | Cost |
|---------------|----------------|------------------|---------------|------|
| `default` | Predefined | StepBack | Generative | 1x |
| `voyager_planning` | **Voyager** | StepBack | Generative | 1.5x |
| `self_refine` | Predefined | **SelfRefine** | None | 2x |
| `cot_sc` | Predefined | **COTSC** (n=5) | None | 5x |
| `voyager_memory` | Predefined | StepBack | **Voyager** | 1.2x |
| `openagi` | **OpenAGI** | StepBack | Generative | 0.8x |
| `hybrid` | **HuggingGPT** | COT | **TP** | 3x |

### Simulation Agent (Track 1)

| Workflow Name | Planning Module | Reasoning Module | Memory Module | Cost |
|---------------|----------------|------------------|---------------|------|
| `default` | Predefined | COT | DILU | 1x |
| `dilu_reasoning` | Predefined | **DILU** | DILU | 1x |
| `self_refine` | Predefined | **SelfRefine** | DILU | 2x |
| `stepback` | Predefined | **StepBack** | DILU | 1.2x |
| `voyager_memory` | Predefined | COT | **Voyager** | 1.2x |
| `openagi` | **OpenAGI** | COT | DILU | 0.8x |
| `hybrid` | **HuggingGPT** | **COTSC** (n=5) | **Generative** | 6x |

### Module Descriptions

**Planning Modules:**
- Predefined: Hardcoded simple plan
- Voyager: Subgoals-based decomposition
- OpenAGI: Minimal todo list
- HuggingGPT: Dependency-aware planning

**Reasoning Modules:**
- StepBack: Extract principles first, then solve
- SelfRefine: Generate, reflect, improve
- COT: Chain-of-thought step-by-step
- COTSC: Self-consistency with 5 generations
- DILU: System prompts for authentic behavior

**Memory Modules:**
- DILU: Basic similarity search
- Generative: Importance-scored retrieval
- Voyager: Summarized trajectories
- TP: Trajectory-based planning
- None: No memory used

## 2. How to Run Evaluations

### Test Recommendation Workflows (Track 2)

```bash
cd example
python test_recommendation_accuracy.py
```

**Default behavior:** Tests 3 workflows (default, self_refine, openagi) on 10 tasks

**Options:**
```bash
# Test more tasks for reliable results
python test_recommendation_accuracy.py --num-tasks 20

# Test specific workflows
python test_recommendation_accuracy.py --workflows default self_refine hybrid

# Test all workflows (expensive!)
python test_recommendation_accuracy.py --workflows all --num-tasks 5

# Test on different dataset
python test_recommendation_accuracy.py --dataset amazon --num-tasks 10
```

**Available workflow names for --workflows:**
`default`, `voyager`, `self_refine`, `cot_sc`, `voyager_memory`, `openagi`, `hybrid`, `all`

**Output:** Shows NDCG@10, Hit Rate@10, Precision@10 for each workflow and tells you which performs best.

### Use Best Workflow in Your Agent

After testing, use the best workflow:

```python
from EnhancedRecommendationAgent import EnhancedRecommendationAgent

class MyAgent(EnhancedRecommendationAgent):
    def workflow(self):
        # Replace with your best workflow
        return self.workflow_with_self_refine()

# Use in simulator
simulator.set_agent(MyAgent)
simulator.run_simulation(number_of_tasks=400)
```

### Workflow Method Names

Call these methods on the agent:

**Recommendation Agent:**
- `agent.workflow()` - default
- `agent.workflow_with_voyager_planning()`
- `agent.workflow_with_self_refine()`
- `agent.workflow_with_cot_sc()`
- `agent.workflow_with_voyager_memory()`
- `agent.workflow_with_openagi_planning()`
- `agent.workflow_hybrid_advanced()`

**Simulation Agent:**
- `agent.workflow()` - default
- `agent.workflow_with_dilu_reasoning()`
- `agent.workflow_with_self_refine()`
- `agent.workflow_with_stepback_reasoning()`
- `agent.workflow_with_voyager_memory()`
- `agent.workflow_with_openagi_planning()`
- `agent.workflow_hybrid_advanced()`

---

**That's it!** Test workflows with the script, see which wins, use that one. 🚀

