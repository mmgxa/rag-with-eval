from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF, SCORE_RUBRIC_TEMPLATE

QA_GEN_PROMPT = """
Your task is to write a single factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""


PROM_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

GROUNDEDNESS_RUBRIC_TEMP = {
    "criteria": "Does the model demonstrate the ability to base its responses clearly and directly on the provided context, ensuring accurate and relevant answers?",
    "score1_description": "The question cannot be answered based on the provided context. It may refer to unrelated or nonexistent concepts.",
    "score2_description": "The question is weakly grounded; the context provides minimal or unclear information for forming an answer.",
    "score3_description": "The question is partially grounded, with some connection to the context, but additional clarification would be required.",
    "score4_description": "The question is mostly grounded, and the answer is clear with minimal interpretation from the context.",
    "score5_description": "The question is fully and clearly grounded in the context, and the answer can be directly derived without ambiguity.",
}


GROUNDEDNESS_RUBRIC = SCORE_RUBRIC_TEMPLATE.format(**GROUNDEDNESS_RUBRIC_TEMP)
# RELEVANCE_RUBRIC = SCORE_RUBRIC_TEMPLATE.format(**RELEVANCE_RUBRIC_TEMP)
# STANDALONE_RUBRIC = SCORE_RUBRIC_TEMPLATE.format(**STANDALONE_RUBRIC_TEMP)


EVALUATION_RUBRIC_TEMP = {
    "criteria": "Is the model proficient in delivering responses that are correct, accurate, and factually aligned with the reference answer?",
    "score1_description": "The response is completely incorrect, inaccurate, and/or not factual.",
    "score2_description": "The response is mostly incorrect, inaccurate, and/or not factual.",
    "score3_description": "The response is somewhat correct, accurate, and/or factual.",
    "score4_description": "The response is mostly correct, accurate, and factual.",
    "score5_description": "The response is completely correct, accurate, and factual.",
}
EVALUATION_RUBRIC = SCORE_RUBRIC_TEMPLATE.format(**EVALUATION_RUBRIC_TEMP)


# =====================================
RAG_SYS_PROMPT = """
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer.
Context:

{context}
"""

RAG_PROMPT = """
Now here is the question you need to answer.

Question: {question}
"""
