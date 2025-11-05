# Alternative Workflows Guide

## Overview

This agent now supports **15 different workflow combinations** exploring different Planning, Reasoning, and Memory modules!

**Module Coverage:**
- Planning: 6/6 modules ✅ (100%)
- Reasoning: 6/7 modules ✅ (86%)
- Memory: 4/4 modules ✅ (100%)

---

## 1. All Available Workflows

### Recommendation Agent (Track 2) - 15 Workflows

| Workflow Name | Planning | Reasoning | Memory | Cost | Notes |
|---------------|----------|-----------|--------|------|-------|
| **ORIGINAL WORKFLOWS** |
| `default` | IO | StepBack | Generative | 💰 1x | Balanced baseline |
| `voyager` | **Voyager** | StepBack | None | 💰 1.2x | Complex decomposition |
| `self_refine` | None | **SelfRefine** | None | 💰💰 2x | Iterative refinement |
| `cot_sc` | None | **COTSC** (n=5) | None | 💰💰💰💰💰 5x | ⚠️ Self-consistency |
| `voyager_memory` | None | StepBack | **Voyager** | 💰 1.2x | Trajectory learning |
| `openagi` | **OpenAGI** | StepBack | None | 💰 0.8x | Fast & efficient |
| `hybrid` | **HuggingGPT** | COT | **TP** | 💰💰💰 3x | Advanced combo |
| **NEW WORKFLOWS** |
| `tot` | None | **TOT** | None | 💰💰💰💰💰 8x | ⚠️ Tree of Thoughts |
| `td` | **TD** | StepBack | None | 💰 1x | Temporal dependencies |
| `deps` | **DEPS** | COT | Generative | 💰💰 2x | 🌟 Multi-hop (recommended!) |
| `all_voyager` | **Voyager** | COT | **Voyager** | 💰💰 2x | Full Voyager stack |
| `dilu_memory` | **HuggingGPT** | StepBack | **DILU** | 💰💰 2x | Alternative memory |
| `simple` | None | **IO** | None | 💰 0.5x | ⚡ Minimal baseline |
| `tot_memory` | None | **TOT** | **TP** | 💰💰💰💰💰+ 8x+ | ⚠️ VERY EXPENSIVE |
| `deps_refine` | **DEPS** | **SelfRefine** | Generative | 💰💰💰 3x | Multi-hop + refine |

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

**Planning Modules (6 total):**
- **IO**: Basic Input-Output planning with predefined steps
- **DEPS**: Multi-hop question decomposition (perfect for recommendations!)
- **TD**: Temporal Dependencies - explicit task ordering
- **Voyager**: Subgoals-based detailed decomposition
- **OpenAGI**: Minimal efficient todo lists
- **HuggingGPT**: Dependency-aware planning with task relationships

**Reasoning Modules (7 total, 6 used):**
- **IO**: Basic Input-Output reasoning (simplest/fastest)
- **COT**: Chain-of-thought step-by-step reasoning
- **COTSC**: Self-consistency with 5 generations (expensive!)
- **TOT**: Tree of Thoughts - 3 paths + 5 votes (very expensive!)
- **DILU**: System prompts for authentic behavior (Track 1 only)
- **SelfRefine**: Generate → reflect → improve iteratively
- **StepBack**: Extract high-level principles first, then apply

**Memory Modules (4 total):**
- **DILU**: Basic trajectory similarity search
- **Generative**: Importance-scored retrieval
- **Voyager**: Summarized trajectories before storage
- **TP**: Trajectory-based planning from past experiences
- **None**: No memory used (for speed/simplicity)

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

**Original:** `default`, `voyager`, `self_refine`, `cot_sc`, `voyager_memory`, `openagi`, `hybrid`

**New:** `tot`, `td`, `deps`, `all_voyager`, `dilu_memory`, `simple`, `tot_memory`, `deps_refine`

**Special:** `all` (tests all 15 workflows - expensive!)

**Output:** Shows NDCG@10, Hit Rate@10, Precision@10 for each workflow and tells you which performs best.

### Recommended Test Combinations

**Quick Comparison (Cheap - $0.30):**
```bash
python test_recommendation_accuracy.py \
  --workflows default simple openagi td deps \
  --num-tasks 15
```

**Quality Focus (Moderate - $0.50):**
```bash
python test_recommendation_accuracy.py \
  --workflows deps all_voyager self_refine dilu_memory \
  --num-tasks 10
```

**Premium Testing (Expensive - $0.80):**
```bash
python test_recommendation_accuracy.py \
  --workflows hybrid deps_refine cot_sc \
  --num-tasks 5
```

**Research Only (Very Expensive - $1.00+):**
```bash
python test_recommendation_accuracy.py \
  --workflows tot tot_memory \
  --num-tasks 3
```

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

**Recommendation Agent (15 workflows):**
- `agent.workflow()` - default
- `agent.workflow_with_voyager_planning()` - voyager
- `agent.workflow_with_self_refine()` - self_refine
- `agent.workflow_with_cot_sc()` - cot_sc
- `agent.workflow_with_voyager_memory()` - voyager_memory
- `agent.workflow_with_openagi_planning()` - openagi
- `agent.workflow_hybrid_advanced()` - hybrid
- `agent.workflow_with_tot_reasoning()` - 🆕 tot
- `agent.workflow_with_td_planning()` - 🆕 td
- `agent.workflow_with_deps_planning()` - 🆕 deps (⭐ recommended!)
- `agent.workflow_all_voyager()` - 🆕 all_voyager
- `agent.workflow_with_dilu_memory()` - 🆕 dilu_memory
- `agent.workflow_simple_efficient()` - 🆕 simple
- `agent.workflow_tot_with_memory()` - 🆕 tot_memory (⚠️ expensive!)
- `agent.workflow_deps_self_refine()` - 🆕 deps_refine

**Simulation Agent:**
- `agent.workflow()` - default
- `agent.workflow_with_dilu_reasoning()`
- `agent.workflow_with_self_refine()`
- `agent.workflow_with_stepback_reasoning()`
- `agent.workflow_with_voyager_memory()`
- `agent.workflow_with_openagi_planning()`
- `agent.workflow_hybrid_advanced()`

