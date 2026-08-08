
# Problem (prompting_baselines)

(a)
Question-only:
* Format=1, Correctness=1: 2
* Format=1, Correctness=0: 203
* Format=0, Correctness=0: 1114

R1-zero:
* Format=1, Correctness=1: 0
* Format=1, Correctness=0: 795
* Format=0, Correctness=0: 524

R1-zero 3-shot:
* Format=1, Correctness=1: 224
* Format=1, Correctness=0: 1050
* Format=0, Correctness=0: 45

Observed 10 examples of format reward 1 and correctness reward 0, and 1/10 of them the model output was actually correct but just not parsed correctly.

Example below:
* Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
* Ground truth: 540
* Completion: He walks at a speed of 5 m/sec. </think> <answer> He would run 3 * 3 = 9 sprints in a week, totaling 9 * 60 = 540 meters. </answer>

Observed 10 examples of format reward 0 and correctness reward 0, and similarly 1/10 of them the model output was correct but not parsed correctly.

Example:
--- Example 1 (index 1) ---
* Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
* Ground truth: 3
* Completion: A robe takes 2 bolts of blue fiber and half that much white fiber. So, a robe takes 2 + (2 / 2) = 3 bolts of fiber in total.</think> <answer> It takes 3 bolts of fiber in total.</answer>

(b) It is not enough to just provide the question to the model (question_only prompt). Many times, the model simply continues the question, and outputs "Remember students don’t have to use any variable to compute in their answers" or "What's the final answer?".

For r1_zero prompts, many times the model doesn't actually think about the question presented to it, but instead some other random GSM8K-style problem. 

The model does the best for r1_zero_three_shot prompts. The model's completion and thinking step is now grounded in the actual question since it was presented with few-shot examples.



