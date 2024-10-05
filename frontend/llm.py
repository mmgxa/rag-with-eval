import boto3
import os

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


def llm_bedrock(model: str, context: str, question: str = None):
    final_sys_prompt = RAG_SYS_PROMPT.format(context=context)
    final_prompt = RAG_PROMPT.format(question=question)
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=os.environ.get("BEDROCK_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("BEDROCK_SECRET_KEY"),
        region_name="us-west-2",
    )

    model_id = {
        "llama3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
        "llama3.1-8b": "meta.llama3-1-8b-instruct-v1:0",
    }

    messages = []
    user_message = f"""<s>[INST]{final_prompt}[/INST]"""
    messages.append(
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    )
    modelid = model_id[model]

    response = bedrock_client.converse(
        modelId=modelid,
        messages=messages,
        system=[{"text": final_sys_prompt}],
        inferenceConfig={"maxTokens": 800, "temperature": 0.7, "topP": 0.7},
    )

    response_text = response["output"]["message"]["content"][0]["text"]

    tokens = {
        "prompt_tokens": response["usage"]["inputTokens"],
        "completion_tokens": response["usage"]["outputTokens"],
        "total_tokens": response["usage"]["totalTokens"],
    }

    return response_text, tokens
