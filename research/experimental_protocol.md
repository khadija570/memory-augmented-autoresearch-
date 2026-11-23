# Experimental Protocol — Memory-Augmented Autoresearch

## Research Question
Can structured experimental memory improve the efficiency and decision-making of autonomous LLM research agents under fixed computational budgets?

## Hypothesis
Structured experimental memory may help autonomous research agents select more informative experiments, avoid repeated failures, and achieve better model validation performance.

## Experimental Setup
- Target model: Mini-GPT (decoder-only Transformer)
- Dataset: TinyStories
- Framework: PyTorch
- Hardware: NVIDIA Tesla T4
- Primary metric: Validation cross-entropy loss
- Final evaluation: Held-out test set

## Baselines
1. Random search
2. Autonomous LLM agent with conventional experiment history
3. Autonomous LLM agent with structured experimental memory

## Controlled Variables
Same model initialization protocol, data splits, evaluator, permitted modifications and experimental budget.

## Evaluation Metrics
- Best validation loss
- Improvement per experiment
- Repeated and failed experiments
- GPU training time
- LLM inference cost

## Limitations
This initial study uses a small model and limited budgets. Results are preliminary.