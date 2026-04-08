# Multi-Agent Simulation of Client-Therapist Dialogues with Large Language Models

## Overview

This repository contains a multi-agent simulation framework for client therapist dialogues in mental health settings. The system uses LLM-based agents and follows core principles of Motivational Interviewing MI. The goal is to generate therapeutic conversations that are coherent, safe and clinically meaningful while remaining reproducible and easy to evaluate.

## Main components

- **Dialogue system**  
  Simulates conversations between therapist and client using configurable LLM backends.  
  Implements a multi agent loop with explicit control over turn-taking, stopping conditions and logging.

- **Questionnaire and background stories**  
  Generates DSM-5-TR based screening questionnaires.  
  Creates background stories conditioned on questionnaire results which then seed the dialogue simulation.

- **Evaluation pipeline**  
  Supports automatic metrics for text quality and diversity, MI-specific metrics for quantitively assessing dialogue's MI quality.  
  Supports LLM based evaluation of coherence, empathy, adherence and conversation level quality.  
  Supports human annotation files for external judgement.

## Project structure

The high level layout is as follows:

```
├── src/
│   ├── dialogue/          # Core dialogue generation
│   ├── evaluate/          # Evaluation framework
│   ├── questionnaire/     # Mental health assessment
│   └── utils/             # Shared utilities
├── data/
│   ├── DSM5-TR/           # Assessment questionnaires
│   ├── MI/                # MI guidelines and datasets
│   └── results/           # Generated content and evaluations
├── human_anno/            # Human annotation tools and outputs
├── result_analysis/       # Analysis scripts and visualizations
└── requirements.txt       # Python dependencies
```

Key directories in `src`:

- `src/dialogue`  
  Core simulation loop and agents  
  `dialogue_system.py` orchestrates multi agent dialogue  
  `therapist.py` implements the therapist agent with MI skills  
  `client.py` implements the client agent and response patterns  

- `src/questionnaire`  
  Generation and processing of DSM-5-TR Level 1 assessments  
  Background story generation conditioned on screening outcomes

- `src/evaluate`  
  Scripts for automatic metrics and LLM based evaluation  
  Support for ablation studies and role adherence checks

## Setup

### Prerequisites

- Python 3.8 or newer  
- Packages listed in `requirements.txt`  
- One of:
  - local models through Ollama  
  - OpenAI compatible API endpoint

### Installation

Clone the repository:

```bash
git clone https://github.com/[YOUR-ANONYMOUS-REPO]/StoryMI-MAS-Dialogue.git
cd StoryMI-MAS-Dialogue
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables for remote LLMs if needed:

```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

Optional setup for Ollama:

```bash
ollama pull model_name
```

## Basic workflow

### Step 1: Generate questionnaires and background stories

Generate questionnaire assessments:

```bash
python -m src.questionnaire.main
```

Generate background stories based on questionnaire results:

```bash
python -m src.questionnaire.story_generation
```

To adjust the number of synthetic users edit the loop in `src/questionnaire/story_generation.py`:

```python
for i in range(1, 1001):
    ...
```

### Step 2: Run dialogue simulations

Create therapeutic dialogues for the configured user range:

```bash
python -m src.dialogue.main
```

### Step 3: Evaluate generated dialogues

Example usage for LLM based evaluation:

```python
from src.evaluate.auto_evaluate import ConversationEvaluator

evaluator = ConversationEvaluator(local_llm=True)
results = evaluator.main(start_index=1, end_index=100, model_name="llama3.1")
```

## Configuration

### Model settings

Edit `src/utils/llm.py` to change:

- default model names  
- local vs remote backends 

### MI code customization

Edit `src/dialogue/mi_code.json` to control:

- therapist techniques: reflection, question, therapist_input  
- client response types: change_talk, sustain_talk, neutral  
- skill definitions and optional examples
