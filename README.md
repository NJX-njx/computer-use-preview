# Computer Use Preview

## Quick Start

This section will guide you through setting up and running the Computer Use Preview model. Follow these steps to get started.

### 1. Installation

**Clone the Repository**

```bash
git clone https://github.com/google/computer-use-preview.git
cd computer-use-preview
```

**Set up Python Virtual Environment and Install Dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Install Playwright and Browser Dependencies**

```bash
# Install system dependencies required by Playwright for Chrome
playwright install-deps chrome

# Install the Chrome browser for Playwright
playwright install chrome
```

### 2. Configuration
You can get started using either the Gemini Developer API or Vertex AI.

#### A. If using the Gemini Developer API:

You need a Gemini API key to use the agent:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Or to add this to your virtual environment:

```bash
echo 'export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"' >> .venv/bin/activate
# After editing, you'll need to deactivate and reactivate your virtual
# environment if it's already active:
deactivate
source .venv/bin/activate
```

Replace `YOUR_GEMINI_API_KEY` with your actual key.

#### B. If using the Vertex AI Client:

You need to explicitly use Vertex AI, then provide project and location to use the agent:

```bash
export USE_VERTEXAI=true
export VERTEXAI_PROJECT="YOUR_PROJECT_ID"
export VERTEXAI_LOCATION="YOUR_LOCATION"
```

Or to add this to your virtual environment:

```bash
echo 'export USE_VERTEXAI=true' >> .venv/bin/activate
echo 'export VERTEXAI_PROJECT="your-project-id"' >> .venv/bin/activate
echo 'export VERTEXAI_LOCATION="your-location"' >> .venv/bin/activate
# After editing, you'll need to deactivate and reactivate your virtual
# environment if it's already active:
deactivate
source .venv/bin/activate
```

Replace `YOUR_PROJECT_ID` and `YOUR_LOCATION` with your actual project and location.

### 3. Running the Tool

The primary way to use the tool is via the `main.py` script.

**General Command Structure:**

```bash
python main.py --mode agent --query "Go to Google and type 'Hello World' into the search bar"
```

**Available Environments:**

You can specify a particular environment with the ```--env <environment>``` flag.  Available options:

- `playwright`: Runs the browser locally using Playwright.
- `browserbase`: Connects to a Browserbase instance.

**Local Playwright**

Runs the agent using a Chrome browser instance controlled locally by Playwright.

```bash
python main.py --mode agent --query="Go to Google and type 'Hello World' into the search bar" --env="playwright"
```

You can also specify an initial URL for the Playwright environment:

```bash
python main.py --mode agent --query="Go to Google and type 'Hello World' into the search bar" --env="playwright" --initial_url="https://www.google.com/search?q=latest+AI+news"
```

**Browserbase**

Runs the agent using Browserbase as the browser backend. Ensure the proper Browserbase environment variables are set:`BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID`.

```bash
python main.py --mode agent --query="Go to Google and type 'Hello World' into the search bar" --env="browserbase"
```

### 4. Collect Early Experience Data (New)

You can collect task-agnostic interaction trajectories for pretraining a behavior prior.

```bash
python main.py --mode collect \
	--env playwright \
	--initial_url "https://www.google.com" \
	--collect_output_dir data/early_experience \
	--collect_episodes 2 \
	--collect_max_steps 20 \
	--collect_whitelist_domains "google.com,wikipedia.org"
```

Artifacts are written under `data/early_experience/episodes/<uuid>/`, including per-step PNG screenshots and an `episode.json` index with metadata and actions.

## Agent CLI

The `main.py` script is the command-line interface (CLI) for running the browser agent.

### Command-Line Arguments

| Argument | Description | Required | Default | Supported Environment(s) |
|-|-|-|-|-|
| `--query` | The natural language query for the browser agent to execute. | Yes | N/A | All |
| `--env` | The computer use environment to use. Must be one of the following: `playwright`, or `browserbase` | No | N/A | All |
| `--initial_url` | The initial URL to load when the browser starts. | No | https://www.google.com | All |
| `--highlight_mouse` | If specified, the agent will attempt to highlight the mouse cursor's position in the screenshots. This is useful for visual debugging. | No | False (not highlighted) | `playwright` |
| `--use_behavior_prior` | Enable a simple behavior prior to validate/adjust actions before execution. | No | False | `agent` |
| `--behavior_prior_ckpt` | Path to the behavior prior checkpoint JSON. | No | `checkpoints/behavior_prior.json` | `agent` |

### Environment Variables

| Variable | Description | Required |
|-|-|-|
| GEMINI_API_KEY | Your API key for the Gemini model. | Yes |
| BROWSERBASE_API_KEY | Your API key for Browserbase. | Yes (when using the browserbase environment) |
| BROWSERBASE_PROJECT_ID | Your Project ID for Browserbase. | Yes (when using the browserbase environment) |

### Train and Use the Behavior Prior (New)

Train a lightweight behavior prior from collected episodes (no heavy ML dependencies required):

```bash
python train/train_behavior_prior.py \
	--episodes_dir data/early_experience \
	--screen_w 1440 --screen_h 900 \
	--out checkpoints/behavior_prior.json
```

Run the agent with the prior enabled:

```bash
python main.py --mode agent --env playwright \
	--query "Open Google and search 'Hello World'" \
	--use_behavior_prior \
	--behavior_prior_ckpt checkpoints/behavior_prior.json
```
