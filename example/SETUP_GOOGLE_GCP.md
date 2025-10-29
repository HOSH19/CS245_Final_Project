# Using Google GCP Credits for AgentSociety Challenge

This guide shows you how to use your $50 Google GCP credits with the enhanced agents.

## 🔑 Getting Your Google API Key

### Step 1: Access Google AI Studio
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account (the one with GCP credits)

### Step 2: Create API Key
1. Click **"Create API Key"**
2. Select your **GCP project** (where you have the $50 credits)
3. Copy the generated API key
4. Save it securely!

### Step 3: Enable Billing
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Billing** → **Link a billing account**
3. Make sure your project with the $50 credits is linked
4. The Gemini API will use these credits automatically

## 💰 Pricing (Using Your Credits)

Google Gemini API pricing (as of 2024):

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|-------|----------------------|------------------------|-----------|
| **gemini-1.5-flash** | $0.075 | $0.30 | Fast, cheap, recommended for testing |
| **gemini-1.5-pro** | $1.25 | $5.00 | Higher quality, production |
| **gemini-1.0-pro** | $0.50 | $1.50 | Balanced |

**With $50 credits** on gemini-1.5-flash:
- Approximately **166M input tokens** + **16M output tokens**
- That's **thousands of simulation tasks**!

## 🚀 Quick Start

### Option A: Using .env File (Recommended - Most Secure)

```bash
# 1. Navigate to example directory
cd example

# 2. Create .env file from template
cp env_template.txt .env

# 3. Edit .env file with your actual API keys
# Open .env in your editor and replace the placeholder values
nano .env  # or use your preferred editor

# 4. Install python-dotenv (if not already installed)
pip install python-dotenv

# 5. Run the test - it will automatically load from .env
python -c "
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
"
```

Your `.env` file should look like this:
```bash
# .env file
GOOGLE_API_KEY=AIzaSyA...your-actual-key-here...
OPENAI_API_KEY=sk-proj-...your-actual-key-here...
```

**Note**: The `.env` file is automatically ignored by git, so your API keys stay secure! 🔒

### Option B: Using Environment Variables

```bash
# Set your Google API key
export GOOGLE_API_KEY="your-google-api-key-here"

# Optional: Set OpenAI key for embeddings
export OPENAI_API_KEY="your-openai-key-here"

# Run the test
cd example
python -c "from GoogleGeminiLLM import GoogleGeminiLLM; ..."
```

### Option C: Direct in Code (Not Recommended for Security)

```python
from GoogleGeminiLLM import GoogleGeminiLLM
from EnhancedSimulationAgent import EnhancedSimulationAgent
from websocietysimulator import Simulator

# Initialize Gemini LLM with your key
llm = GoogleGeminiLLM(
    api_key="your-google-api-key-here",
    model="gemini-1.5-flash"  # Fast and cheap
)

# Setup simulator
simulator = Simulator(data_dir="../data_processed", cache=True)
simulator.set_task_and_groundtruth(
    task_dir="./track1/yelp/tasks",
    groundtruth_dir="./track1/yelp/groundtruth"
)

# Use your agent with Gemini
simulator.set_agent(EnhancedSimulationAgent)
simulator.set_llm(llm)

# Run
outputs = simulator.run_simulation(number_of_tasks=10)
results = simulator.evaluate()
print(results)
```

## 📦 Installation

Install the required packages:

```bash
# Install Google Generative AI and python-dotenv
pip install google-generativeai python-dotenv

# Or if using poetry
poetry add google-generativeai python-dotenv
```

## 🎯 Recommended Model Choice

### For Development & Testing
```python
llm = GoogleGeminiLLM(
    api_key=your_key,
    model="gemini-1.5-flash"
)
```
- **Fast**: Responses in < 1 second
- **Cheap**: Uses very little of your credits
- **Good quality**: More than sufficient for testing

### For Final Evaluation
```python
llm = GoogleGeminiLLM(
    api_key=your_key,
    model="gemini-1.5-pro"
)
```
- **Higher quality**: Better reasoning
- **More expensive**: Use after testing with flash
- **Slower**: But more thorough responses

## 🔧 About Embeddings

The agents use embeddings for the memory module. You have two options:

### Option 1: Use OpenAI Embeddings (Recommended)
- **Cost**: ~$0.013 per 1M tokens (very cheap!)
- **Quality**: Excellent
- **Setup**: Get a small OpenAI credit (~$5) for embeddings only

```bash
export OPENAI_API_KEY="your-openai-key"
```

### Option 2: Skip Embeddings
The agents will work without embeddings, but memory features will be limited.

## 💡 Cost Saving Tips

1. **Start with Flash**: Use `gemini-1.5-flash` for all testing
2. **Test Small**: Use `--num-tasks 5` first, then scale up
3. **Use Threading Wisely**: Threading is faster but uses more tokens
4. **Monitor Usage**: Check your [GCP Console](https://console.cloud.google.com/) regularly
5. **Optimize Prompts**: Shorter prompts = fewer tokens = lower cost

## 📊 Estimated Costs

For the competition with **gemini-1.5-flash**:

| Task | Tasks | Est. Cost | % of $50 Credits |
|------|-------|-----------|------------------|
| Quick Test | 5 | ~$0.10 | 0.2% |
| Small Test | 50 | ~$1.00 | 2% |
| Full Track 1 | 400 | ~$8.00 | 16% |
| Full Track 2 | 400 | ~$6.00 | 12% |
| **Both Tracks** | **800** | **~$14** | **28%** |

You'll have **plenty of credits** left for experimentation! 🎉

## 🐛 Troubleshooting

### "API key not valid"
- Make sure you copied the entire key
- Check it's linked to your GCP project with credits
- Try generating a new key

### "Quota exceeded"
- Check your [GCP Billing](https://console.cloud.google.com/billing)
- Make sure credits are applied to the right project
- Enable the Gemini API in your project

### "Module not found: google.generativeai"
```bash
pip install google-generativeai
```

### "No embedding model available"
This is a warning, not an error. The agent will work but memory features are limited.

**Solution**: Get a small OpenAI credit for embeddings:
1. Go to [OpenAI](https://platform.openai.com/)
2. Add $5 to your account
3. Get API key and set: `export OPENAI_API_KEY="..."`

## 📈 Monitoring Your Usage

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Billing** → **Reports**
3. Filter by **Gemini API**
4. Track your spending against the $50 credit

## ✅ Complete Example

```bash
# 1. Install required package
pip install google-generativeai

# 2. Set your API key
export GOOGLE_API_KEY="your-key-from-ai-studio"

# 3. Test with a few tasks
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
results = simulator.evaluate()
print(f'Results: {results}')
"

# 4. If it works, run full evaluation
# (Update test_enhanced_agents.py to support --llm google flag)
```

## 🎓 Next Steps

1. **Get your API key** from Google AI Studio
2. **Install** `google-generativeai` package
3. **Test** with 5 tasks using `gemini-1.5-flash`
4. **Scale up** once you confirm it works
5. **Monitor** your usage in GCP Console

Your $50 credits should be more than enough for the entire competition! 🚀

## 📞 Resources

- **Google AI Studio**: https://aistudio.google.com/
- **Gemini API Docs**: https://ai.google.dev/docs
- **GCP Console**: https://console.cloud.google.com/
- **Pricing**: https://ai.google.dev/pricing

