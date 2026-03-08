"""
Generate matched critique and paraphrase stimuli for GSM8K questions using GPT-4.1 API.
Each question gets: (1) an initial answer, (2) a self-critique + revision, (3) a paraphrase control.
"""

import json
import os
import random
import time
from openai import OpenAI

random.seed(42)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

def load_gsm8k(n=150):
    """Load n questions from GSM8K test set."""
    from datasets import load_from_disk
    ds = load_from_disk("datasets/gsm8k")
    questions = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds["test"]]
    random.shuffle(questions)
    return questions[:n]


def extract_numeric_answer(answer_text):
    """Extract the final numeric answer from GSM8K format (#### NUMBER)."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip().split("\n")[-1].strip()


def generate_stimuli_for_question(question, answer, idx):
    """Generate critique and paraphrase stimuli for a single question."""
    numeric_ans = extract_numeric_answer(answer)

    # Decide if the initial answer should be wrong (50%) for probe analysis
    make_wrong = random.random() < 0.5

    prompt = f"""You are helping create experimental stimuli for a research study on self-critique in language models.

Given this math question and its correct answer, generate THREE things:

1. **initial_answer**: A step-by-step solution that {"arrives at a WRONG answer (make a plausible arithmetic or reasoning error)" if make_wrong else "arrives at the CORRECT answer"}. Keep it 2-4 sentences.

2. **critique**: A self-critique of the initial answer (as if the model is reviewing its own work). Start with "Wait, let me reconsider." Then identify what's right/wrong and provide a corrected answer. Keep it 2-4 sentences.

3. **paraphrase**: A paraphrase/restatement of the initial answer that says the same thing in different words WITHOUT any critique or correction. Start with "In other words," and keep it 2-4 sentences, similar length to the critique.

Question: {question}
Correct answer: {numeric_ans}

Respond in JSON format:
{{"initial_answer": "...", "critique": "...", "paraphrase": "...", "initial_is_correct": {str(not make_wrong).lower()}, "final_answer_after_critique": "..."}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result["question"] = question
        result["correct_answer"] = numeric_ans
        result["ground_truth_solution"] = answer
        result["index"] = idx
        return result
    except Exception as e:
        print(f"  Error for question {idx}: {e}")
        return None


def main():
    print("Loading GSM8K questions...")
    questions = load_gsm8k(150)
    print(f"Loaded {len(questions)} questions")

    stimuli = []
    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"Generating stimuli {i}/{len(questions)}...")
        result = generate_stimuli_for_question(q["question"], q["answer"], i)
        if result:
            stimuli.append(result)
        # Small delay to avoid rate limits
        if i % 20 == 0 and i > 0:
            time.sleep(1)

    output_path = "results/data/stimuli.json"
    with open(output_path, "w") as f:
        json.dump(stimuli, f, indent=2)

    print(f"\nGenerated {len(stimuli)} stimuli, saved to {output_path}")
    print(f"  Correct initial answers: {sum(1 for s in stimuli if s.get('initial_is_correct', False))}")
    print(f"  Wrong initial answers: {sum(1 for s in stimuli if not s.get('initial_is_correct', False))}")

    return stimuli


if __name__ == "__main__":
    main()
