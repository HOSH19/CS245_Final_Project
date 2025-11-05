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

# Alternative Workflow Structure Guide

## Quick Reference: When to Use Each Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WORKFLOW DECISION TREE                            │
└─────────────────────────────────────────────────────────────────────┘

Need best quality? → workflow_hybrid_advanced()
                     (Most expensive: 3 modules, most API calls)

Need reliability?  → workflow_with_cot_sc() 
                     (Expensive: 5 reasoning attempts)

Need efficiency?   → workflow_with_openagi_planning()
                     (Minimal planning, efficient execution)

Need complex decomposition? → workflow_with_voyager_planning()
                              (Detailed subgoal planning)

Need refinement?   → workflow_with_self_refine()
                     (Initial + refined output)

Need memory?       → workflow_with_voyager_memory()
                     (Summarized trajectory memory)

Balanced approach? → workflow() [DEFAULT]
                     (Good balance of all aspects)
```

---

## Module Combinations Matrix

| Workflow | Planning | Reasoning | Memory | Cost | Quality |
|----------|----------|-----------|--------|------|---------|
| **workflow()** (default) | IO | StepBack | Generative | 💰 Medium | ⭐⭐⭐ Good |
| **workflow_with_voyager_planning()** | Voyager | StepBack | None | 💰💰 Medium-High | ⭐⭐⭐ Good |
| **workflow_with_self_refine()** | None | SelfRefine | None | 💰💰 Medium-High | ⭐⭐⭐⭐ Better |
| **workflow_with_cot_sc()** | None | COTSC (x5) | None | 💰💰💰💰 Very High | ⭐⭐⭐⭐⭐ Excellent |
| **workflow_with_voyager_memory()** | None | StepBack | Voyager | 💰💰 Medium-High | ⭐⭐⭐⭐ Better |
| **workflow_with_openagi_planning()** | OpenAGI | StepBack | None | 💰 Medium | ⭐⭐⭐ Good |
| **workflow_hybrid_advanced()** | HuggingGPT | COT | TP | 💰💰💰💰💰 Highest | ⭐⭐⭐⭐⭐ Best |

---

## Detailed Workflow Structures

### 1. workflow() - **DEFAULT / BALANCED**

```
┌─────────────────────────────────────┐
│         DEFAULT WORKFLOW            │
├─────────────────────────────────────┤
│ Planning:   PlanningIO              │
│   ↓ Creates predefined plan         │
│                                     │
│ Memory:     MemoryGenerative        │
│   ↓ Stores 30 reviews               │
│                                     │
│ Reasoning:  ReasoningStepBack       │
│   ↓ Extract principles → Apply      │
│                                     │
│ Output:     Ranked item list        │
└─────────────────────────────────────┘

When to use:
✓ General purpose recommendations
✓ Balanced quality and cost
✓ Unknown user preferences
✓ Standard recommendation tasks
```

---

### 2. workflow_with_voyager_planning() - **COMPLEX DECOMPOSITION**

```
┌─────────────────────────────────────┐
│      VOYAGER PLANNING WORKFLOW      │
├─────────────────────────────────────┤
│ Planning:   PlanningVoyager         │
│   ↓ Detailed subgoal decomposition  │
│   ↓ Multiple structured subtasks    │
│                                     │
│ Memory:     None                    │
│                                     │
│ Reasoning:  ReasoningStepBack       │
│   ↓ Extract principles → Apply      │
│                                     │
│ Output:     Ranked item list        │
└─────────────────────────────────────┘

When to use:
✓ Complex recommendation scenarios
✓ Multi-step reasoning needed
✓ Tasks requiring careful planning
✓ When task dependencies matter
```

---

### 3. workflow_with_self_refine() - **ITERATIVE IMPROVEMENT**

```
┌─────────────────────────────────────┐
│      SELF-REFINE WORKFLOW           │
├─────────────────────────────────────┤
│ Planning:   None (Simple)           │
│                                     │
│ Memory:     None                    │
│                                     │
│ Reasoning:  ReasoningSelfRefine     │
│   ↓ Generate initial ranking        │
│   ↓ Reflect on quality              │
│   ↓ Refine and improve              │
│                                     │
│ Output:     Refined item list       │
└─────────────────────────────────────┘

When to use:
✓ Quality is more important than cost
✓ Initial results need improvement
✓ Complex preference patterns
✓ When refinement helps accuracy
```

---

### 4. workflow_with_cot_sc() - **MAXIMUM RELIABILITY**

```
┌─────────────────────────────────────┐
│    COT SELF-CONSISTENCY WORKFLOW    │
├─────────────────────────────────────┤
│ Planning:   None (Simple)           │
│                                     │
│ Memory:     None                    │
│                                     │
│ Reasoning:  ReasoningCOTSC          │
│   ↓ Generate 5 reasoning paths      │
│   ↓ Each with chain-of-thought      │
│   ↓ Select most consistent answer   │
│                                     │
│ Output:     Consensus-based list    │
└─────────────────────────────────────┘

When to use:
✓ Reliability is critical
✓ Cost is not a concern (5x API calls!)
✓ Need confidence in results
✓ High-stakes recommendations

⚠️ WARNING: 5x more expensive!
```

---

### 5. workflow_with_voyager_memory() - **TRAJECTORY LEARNING**

```
┌─────────────────────────────────────┐
│      VOYAGER MEMORY WORKFLOW        │
├─────────────────────────────────────┤
│ Planning:   None (Simple)           │
│                                     │
│ Memory:     MemoryVoyager           │
│   ↓ Summarize 20 review trajectories│
│   ↓ Store concise patterns          │
│   ↓ Retrieve relevant context       │
│                                     │
│ Reasoning:  ReasoningStepBack       │
│   ↓ Enhanced with memory context    │
│                                     │
│ Output:     Memory-informed list    │
└─────────────────────────────────────┘

When to use:
✓ Learning from review patterns
✓ Users with extensive history
✓ Pattern recognition important
✓ Long-term preference modeling
```

---

### 6. workflow_with_openagi_planning() - **EFFICIENT & STREAMLINED**

```
┌─────────────────────────────────────┐
│      OPENAGI PLANNING WORKFLOW      │
├─────────────────────────────────────┤
│ Planning:   PlanningOPENAGI         │
│   ↓ Minimal, efficient todo list    │
│   ↓ Only essential steps            │
│                                     │
│ Memory:     None                    │
│                                     │
│ Reasoning:  ReasoningStepBack       │
│   ↓ Extract principles → Apply      │
│                                     │
│ Output:     Ranked item list        │
└─────────────────────────────────────┘

When to use:
✓ Need efficiency over complexity
✓ Simple recommendation tasks
✓ Cost optimization important
✓ Fast turnaround required
```

---

### 7. workflow_hybrid_advanced() - **MAXIMUM QUALITY**

```
┌─────────────────────────────────────┐
│      HYBRID ADVANCED WORKFLOW       │
├─────────────────────────────────────┤
│ Planning:   PlanningHUGGINGGPT      │
│   ↓ Dependency-aware planning       │
│   ↓ Task relationship modeling      │
│                                     │
│ Memory:     MemoryTP                │
│   ↓ Trajectory-based planning       │
│   ↓ Store 15 review trajectories    │
│   ↓ Extract trajectory insights     │
│                                     │
│ Reasoning:  ReasoningCOT            │
│   ↓ Step-by-step chain-of-thought   │
│   ↓ Memory-enhanced reasoning       │
│                                     │
│ Output:     Best quality list       │
└─────────────────────────────────────┘

When to use:
✓ Maximum quality needed
✓ Cost is not a concern
✓ Complex user preferences
✓ Research / benchmarking
✓ High-value recommendations

⚠️ WARNING: Most expensive option!
Uses 3 sophisticated modules
```

---

## Module Descriptions

### **Planning Modules**

| Module | Description | Best For |
|--------|-------------|----------|
| **PlanningIO** | Input-Output planning with predefined steps | General tasks |
| **PlanningVoyager** | Subgoal-based detailed decomposition | Complex tasks |
| **PlanningOPENAGI** | Minimal efficient todo lists | Simple/fast tasks |
| **PlanningHUGGINGGPT** | Dependency-aware planning | Tasks with dependencies |

### **Reasoning Modules**

| Module | Description | Cost | Quality |
|--------|-------------|------|---------|
| **ReasoningStepBack** | High-level principles → Application | 💰 | ⭐⭐⭐ |
| **ReasoningSelfRefine** | Generate → Reflect → Refine | 💰💰 | ⭐⭐⭐⭐ |
| **ReasoningCOTSC** | 5 reasoning paths → Consensus | 💰💰💰💰💰 | ⭐⭐⭐⭐⭐ |
| **ReasoningCOT** | Step-by-step chain-of-thought | 💰 | ⭐⭐⭐ |

### **Memory Modules**

| Module | Description | Best For |
|--------|-------------|----------|
| **MemoryGenerative** | Stores and retrieves patterns | General memory needs |
| **MemoryVoyager** | Summarizes trajectories before storage | Concise retrieval |
| **MemoryTP** | Trajectory-based planning memory | Learning from sequences |

---

## Common Workflow Execution Pattern

All workflows now follow this clean pattern:

```python
def workflow_with_MODULE_NAME(self) -> List[str]:
    # 1. Initialize specific module(s)
    module = ModuleClass(llm=self.llm, ...)
    
    # 2. Call generic workflow with configuration
    return self._execute_generic_workflow(
        workflow_name="Descriptive Name",
        planning_module=module or None,
        reasoning_module=module or self.reasoning,
        memory_module=module or None
    )
```

### Behind the scenes, `_execute_generic_workflow()` does:

```
1. Create plan (if planning_module provided)
   └─> Generates structured subtasks

2. Gather user data
   └─> _gather_user_data()
       ├─> Get user profile
       └─> Get user reviews

3. Handle memory (if memory_module provided)
   └─> _store_reviews_in_memory()
       ├─> Store up to 20 reviews
       └─> Retrieve relevant context

4. Gather candidate items
   └─> _gather_candidate_items()
       ├─> Fetch each item
       ├─> Filter relevant fields
       └─> Handle errors gracefully

5. Generate recommendations
   └─> _generate_recommendations_with_reasoning()
       ├─> Analyze user preferences
       ├─> Create comprehensive prompt
       ├─> Call reasoning module
       ├─> Parse result
       └─> Validate output

6. Return validated list
```

---

## API Cost Comparison

Assuming 1 task with 20 items and 50 user reviews:

| Workflow | Approx. API Calls | Relative Cost |
|----------|-------------------|---------------|
| **workflow_with_openagi_planning()** | ~3-4 | 💰 Baseline (1x) |
| **workflow()** (default) | ~4-5 | 💰 1.2x |
| **workflow_with_voyager_planning()** | ~4-5 | 💰 1.2x |
| **workflow_with_voyager_memory()** | ~5-6 | 💰💰 1.5x |
| **workflow_with_self_refine()** | ~6-7 | 💰💰 1.8x |
| **workflow_hybrid_advanced()** | ~7-9 | 💰💰💰 2.2x |
| **workflow_with_cot_sc()** | ~15-20 | 💰💰💰💰💰 5x |

---

## Quick Selection Guide

**Choose based on priority:**

```
Priority: QUALITY
  → workflow_hybrid_advanced() or workflow_with_cot_sc()

Priority: EFFICIENCY  
  → workflow_with_openagi_planning() or workflow() (default)

Priority: RELIABILITY
  → workflow_with_cot_sc()

Priority: LEARNING PATTERNS
  → workflow_with_voyager_memory()

Priority: BALANCED
  → workflow() (default)

Priority: ITERATIVE IMPROVEMENT
  → workflow_with_self_refine()

Priority: COMPLEX TASKS
  → workflow_with_voyager_planning()
```

---

## Testing Different Workflows

```python
# Example: Compare different workflows
from websocietysimulator import Simulator
from EnhancedRecommendationAgent import EnhancedRecommendationAgent

simulator = Simulator(data_dir="./data_processed", cache=True)
simulator.set_task_and_groundtruth(
    task_dir="./track2/yelp/tasks",
    groundtruth_dir="./track2/yelp/groundtruth"
)

# Test different workflows
agent = EnhancedRecommendationAgent(llm=llm)

# Method 1: Override the workflow method
agent.workflow = agent.workflow_with_voyager_planning
outputs = simulator.run_simulation(number_of_tasks=5)

# Method 2: Call workflow directly on agent instance
# (after simulator assigns task to agent)
recommendations = agent.workflow_with_self_refine()
```

