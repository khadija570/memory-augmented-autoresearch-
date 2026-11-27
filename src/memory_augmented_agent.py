
import json

from src.research_agent import ResearchAgent
from src.experiment_runner import SEARCH_SPACE


class MemoryAugmentedResearchAgent(ResearchAgent):
    """
    Research agent that uses structured experimental
    memory to generate new research hypotheses.
    """

    def __init__(
        self,
        memory_manager,
        model_name="gemini-3.8-flash"
    ):

        super().__init__(model_name=model_name)

        self.memory_manager = memory_manager

    def build_prompt(self, history, baseline_config):
        """
        Build a research prompt using structured memory
        instead of chronological experimental history.
        """

        memory_context = self.memory_manager.get_context()

        return f"""
You are an autonomous machine learning research agent.

RESEARCH OBJECTIVE:
Minimize the validation cross-entropy loss of a
Mini-GPT language model trained on TinyStories.

ALLOWED SEARCH SPACE:
{json.dumps(SEARCH_SPACE, indent=2)}

BASELINE CONFIGURATION:
{json.dumps(baseline_config, indent=2)}

STRUCTURED EXPERIMENTAL MEMORY:
{json.dumps(memory_context, indent=2)}

MEMORY INTERPRETATION:

1. Research state:
   Identify the current best experimental configuration.

2. Episodic memory:
   Analyze the best and recent experimental results.

3. Semantic memory:
   Use controlled experimental comparisons to identify
   promising research directions.

4. Failure memory:
   Learn from valid negative experimental observations.
   Do not interpret operational failures as evidence
   about hyperparameter performance.

5. Evidence quality:
   Semantic findings are preliminary observations,
   not universally established scientific conclusions.
   Consider the experimental context before applying
   any finding to a new configuration.

EXPERIMENTAL CONSTRAINTS:

- Training budget: 300 optimization steps.
- Keep the model architecture unchanged.
- Keep the dataset and data splits unchanged.
- Keep the random seed unchanged.
- Keep the evaluation procedure unchanged.
- Modify only the allowed hyperparameters.
- Do not invent experimental results.
- Avoid previously tested configurations.

YOUR TASK:

Analyze the available experimental evidence.

Identify a promising research direction.

Formulate one testable hypothesis.

Propose one new hyperparameter configuration.

Explain why this experiment is worth conducting.

Balance exploration of new configurations with
exploitation of promising previous observations.

Return only a valid JSON object with this structure:

{{
    "hypothesis": "A testable scientific hypothesis",
    "reasoning": "Evidence-backed justification",
    "parameters": {{
        "learning_rate": 0.001,
        "dropout": 0.1,
        "weight_decay": 0.01
    }}
}}

The parameter values above are only an example.
Select values based on the experimental evidence.
"""

    def propose_from_memory(self, baseline_config):
        """
        Generate and validate a memory-informed proposal.
        """

        proposal = super().propose(
            history=[],
            baseline_config=baseline_config
        )

        parameters = proposal["parameters"]

        # Check the complete history, not only
        # the experiments included in retrieved memory.
        if self.memory_manager.has_tested(parameters):

            raise ValueError(
                "Duplicate configuration: "
                "already tested in valid experiments"
            )

        return proposal
