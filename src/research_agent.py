
import os
import json

from google import genai
from google.genai import types

from src.experiment_runner import (
    SEARCH_SPACE,
    validate_proposal,
)


class ResearchAgent:

    def __init__(self, model_name="gemini-3.8-flash"):

        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        self.model_name = model_name

    def build_prompt(self, history, baseline_config):

        return f"""
You are an autonomous machine learning researcher.

RESEARCH OBJECTIVE:
Minimize the validation cross-entropy loss
of a Mini-GPT model trained on TinyStories.

ALLOWED SEARCH SPACE:
{json.dumps(SEARCH_SPACE, indent=2)}

BASELINE CONFIGURATION:
{json.dumps(baseline_config, indent=2)}

EXPERIMENT HISTORY:
{json.dumps(history, indent=2)}

FIXED CONDITIONS:
- 300 training steps
- Fixed training seed
- Fixed model architecture
- Fixed validation batches
- Fixed dataset
- Fixed evaluation metric

YOUR TASK:
1. Analyze previous experimental results.
2. Identify a promising research direction.
3. Formulate a testable hypothesis.
4. Propose one new configuration.
5. Explain why this experiment is useful.

RULES:
- Never modify the evaluation procedure.
- Stay within the allowed search space.
- Avoid repeating identical configurations.
- Prefer informative experiments.
- Do not invent experimental results.
- Do not assume that a hypothesis is proven.

Return only a JSON object with this structure:

{{
    "hypothesis": "Your research hypothesis",
    "reasoning": "Your experimental justification",
    "parameters": {{
        "learning_rate": 0.001,
        "dropout": 0.1,
        "weight_decay": 0.01
    }}
}}
"""

    def propose(self, history, baseline_config):

        prompt = self.build_prompt(
            history,
            baseline_config
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        proposal = json.loads(response.text)

        for field in ["hypothesis", "reasoning"]:
            if not isinstance(proposal.get(field), str):
                raise ValueError(
                    f"Invalid or missing {field}"
                )

        proposal["parameters"] = validate_proposal(
            proposal["parameters"]
        )

        return proposal
