# Testing with Google Gemini - Quick Start Guide

This guide explains how to use the Gemini test scripts (`test_gemini_rec.py` and `test_gemini_simulation.py`) to quickly validate your setup and run simulations with Google's Gemini API.

## 📋 Overview

These test scripts provide a simple way to:
- **`test_gemini_simulation.py`**: Test the **Track 1 (User Modeling)** agent with Gemini LLM
- **`test_gemini_rec.py`**: Test the **Track 2 (Recommendation)** agent with Gemini LLM

Both scripts run a small number of tasks (5 by default) to verify your setup is working correctly before running full evaluations.

## 🎯 What Do These Scripts Do?

### `test_gemini_simulation.py` - Track 1 Testing
- Uses `EnhancedSimulationAgent` for user behavior modeling
- Runs on **Track 1** tasks (predicting user interactions)
- Tests against Yelp dataset by default
- Evaluates how well the agent predicts user ratings

### `test_gemini_rec.py` - Track 2 Testing
- Uses `EnhancedRecommendationAgent` for personalized recommendations
- Runs on **Track 2** tasks (recommending items to users)
- Tests against Yelp dataset by default
- Evaluates recommendation quality (NDCG, Hit Rate, etc.)

## 🔧 Prerequisites

### 1. Install Required Packages

```bash
pip install google-generativeai python-dotenv
```

Or if using Poetry:
```bash
poetry add google-generativeai python-dotenv
```

### 2. Get Your Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Select your GCP project (with your $50 credits)
4. Copy the API key

> 📖 For detailed setup instructions, see [`README_SETUP_GOOGLE_GCP.md`](./README_SETUP_GOOGLE_GCP.md)

### 3. Set Up Environment Variables

**Option A: Using .env file (Recommended)**

Create a `.env` file in the `example/` directory:

```bash
# Create .env file
cd example
cat > .env << EOF
GOOGLE_API_KEY=your-google-api-key-here
OPENAI_API_KEY=your-openai-key-here  # Optional, for embeddings
EOF
```

**Option B: Export environment variables**

```bash
export GOOGLE_API_KEY="your-google-api-key-here"
export OPENAI_API_KEY="your-openai-key-here"  # Optional
```

### 4. Prepare Data

Make sure you have processed the data first.
This refers to the `data_processed/` directory with `item.json`, `review.json`, and `user.json`.

## 🚀 Running the Tests

### Test Track 1 (User Modeling)

```bash
cd example
python test_gemini_simulation.py
```

**What it does:**
1. Initializes Gemini LLM with `gemini-2.0-flash` model
2. Loads the simulator with cached data
3. Sets up Track 1 tasks from `./track1/yelp/tasks`
4. Runs the `EnhancedSimulationAgent` on 5 tasks
5. Evaluates and prints results

**Expected output:**
```
Loading data...
Running simulation on 5 tasks...
Task 1/5: [User interaction prediction]
Task 2/5: [User interaction prediction]
...
{'mae': 0.45, 'rmse': 0.67, 'accuracy': 0.82}
```

### Test Track 2 (Recommendation)

```bash
cd example
python test_gemini_rec.py
```

**What it does:**
1. Initializes Gemini LLM with `gemini-2.0-flash` model
2. Loads the simulator with cached data
3. Sets up Track 2 tasks from `./track2/yelp/tasks`
4. Runs the `EnhancedRecommendationAgent` on 5 tasks
5. Evaluates and prints results

**Expected output:**
```
Loading data...
Running simulation on 5 tasks...
Task 1/5: [Generating recommendations]
Task 2/5: [Generating recommendations]
...
{'ndcg@10': 0.45, 'hit_rate@10': 0.65, 'precision@10': 0.35}
```

## 📝 Understanding the Results

### Track 1 (Simulation) Metrics
- **MAE (Mean Absolute Error)**: Average difference between predicted and actual ratings (lower is better)
- **RMSE (Root Mean Squared Error)**: Root of squared differences (lower is better)
- **Accuracy**: Percentage of correct predictions (higher is better)

### Track 2 (Recommendation) Metrics
- **NDCG@10**: Normalized Discounted Cumulative Gain (0-1, higher is better)
- **Hit Rate@10**: Percentage of tasks with at least one correct item in top 10 (higher is better)
- **Precision@10**: Percentage of recommended items that are relevant (higher is better)

## 🎛️ Customizing the Tests

You can modify the test scripts to:

### Change the Number of Tasks

```python
# Instead of 5 tasks
outputs = simulator.run_simulation(number_of_tasks=5)

# Run 20 tasks
outputs = simulator.run_simulation(number_of_tasks=20)
```

### Change the Model

```python
# Use faster model (default)
llm = GoogleGeminiLLM(model='gemini-2.0-flash')

# Use more powerful model
llm = GoogleGeminiLLM(model='gemini-1.5-pro')

# Use older model
llm = GoogleGeminiLLM(model='gemini-1.5-flash')
```

### Test Different Datasets

```python
# Test with Amazon data
simulator.set_task_and_groundtruth(
    task_dir='./track1/amazon/tasks',
    groundtruth_dir='./track1/amazon/groundtruth'
)

# Test with Goodreads data
simulator.set_task_and_groundtruth(
    task_dir='./track1/goodreads/tasks',
    groundtruth_dir='./track1/goodreads/groundtruth'
)
```

### Disable Caching

```python
# Without caching (slower, but uses less memory)
simulator = Simulator(data_dir='../data_processed', cache=False)
```

## 💰 Cost Estimates

Using `gemini-2.0-flash` (recommended for testing):

| Test | Tasks | Estimated Cost | Time |
|------|-------|----------------|------|
| Quick Test (5 tasks) | 5 | ~$0.05-0.10 | 1-2 min |
| Medium Test (20 tasks) | 20 | ~$0.20-0.40 | 5-10 min |
| Full Test (50 tasks) | 50 | ~$0.50-1.00 | 15-30 min |

> 💡 **Tip**: Always test with 5-10 tasks first to validate your setup before running larger evaluations!

## 🐛 Troubleshooting

### Error: "GOOGLE_API_KEY not found"

**Solution:**
```bash
# Check if your .env file exists
ls -la .env

# Or set the environment variable directly
export GOOGLE_API_KEY="your-key-here"
```

### Error: "Module 'google.generativeai' not found"

**Solution:**
```bash
pip install google-generativeai
```

### Error: "No such file or directory: '../data_processed'"

**Solution:**
```bash
# Go to root directory and process data
cd ..
python data_process.py
cd example
```

### Warning: "No embedding model available"

This is not an error! The agents will work, but with limited memory features.

**To enable embeddings (optional):**
```bash
# Get OpenAI API key from https://platform.openai.com/
export OPENAI_API_KEY="your-openai-key"
```

### Error: "Response blocked by safety filters"

**Solution:** The script already has relaxed safety settings. If you still see this:
1. Try rephrasing your prompts
2. Check the task data for problematic content
3. Use a different task file

### Slow Performance

**Solutions:**
- Ensure `cache=True` in Simulator initialization
- Use `gemini-2.0-flash` instead of `gemini-1.5-pro`
- Reduce the number of tasks for testing
- Check your internet connection

## 📊 Next Steps After Testing

Once your tests pass successfully:

### 1. Run Full Evaluations

Create a more comprehensive test script:

```python
from GoogleGeminiLLM import GoogleGeminiLLM
from websocietysimulator import Simulator
from EnhancedSimulationAgent import EnhancedSimulationAgent

llm = GoogleGeminiLLM(model='gemini-2.0-flash')
simulator = Simulator(data_dir='../data_processed', cache=True)

# Run on all three datasets
datasets = ['yelp', 'amazon', 'goodreads']
for dataset in datasets:
    print(f"\n=== Testing {dataset} ===")
    simulator.set_task_and_groundtruth(
        task_dir=f'./track1/{dataset}/tasks',
        groundtruth_dir=f'./track1/{dataset}/groundtruth'
    )
    outputs = simulator.run_simulation(number_of_tasks=50)
    results = simulator.evaluate()
    print(f"{dataset} results: {results}")
```

### 2. Optimize Your Agent

- Review the enhanced agent code in `EnhancedSimulationAgent.py` or `EnhancedRecommendationAgent.py`
- Experiment with different prompts
- Adjust agent parameters
- Test with different LLM models

### 3. Create Your Submission

See the main README for submission guidelines:
- Track 1: Submit your modeling agent
- Track 2: Submit your recommendation agent
- Package according to submission format

## 📚 Additional Resources

- **Setup Guide**: [`README_SETUP_GOOGLE_GCP.md`](./README_SETUP_GOOGLE_GCP.md) - Detailed GCP setup
- **Agent Documentation**: [`README_ENHANCED_AGENTS.md`](./README_ENHANCED_AGENTS.md) - Understanding the enhanced agents
- **Google Gemini Docs**: https://ai.google.dev/docs - API documentation
- **Main Challenge README**: [`../README.md`](../README.md) - Full competition details

## 🎓 Quick Reference

### Test Track 1
```bash
cd example
export GOOGLE_API_KEY="your-key"
python test_gemini_simulation.py
```

### Test Track 2
```bash
cd example
export GOOGLE_API_KEY="your-key"
python test_gemini_rec.py
```

### Test Both Tracks
```bash
cd example
export GOOGLE_API_KEY="your-key"
python test_gemini_simulation.py && python test_gemini_rec.py
```

## ❓ Questions?

- Check the main documentation in the repository
- Review the enhanced agents code for implementation details
- See `README_SETUP_GOOGLE_GCP.md` for API key and billing issues

---

**Happy Testing! 🚀**

With these test scripts, you can quickly validate your setup and ensure everything works before running full evaluations. Start small (5 tasks), verify the results, then scale up!