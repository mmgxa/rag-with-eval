import boto3
import os
from dotenv import load_dotenv
from .config import get_project_root
from pydantic import BaseModel
from typing import Literal, List
import openai


from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT_WO_REF, ABSOLUTE_PROMPT

load_dotenv(get_project_root() + "/.env")


class Providers(BaseModel):
    provider: Literal["bedrock", "together"]


class LLM:
    def __init__(self, model, provider: str):

        self.provider_ = Providers(provider=provider).provider

        self.model_ = model

    def generate(self, prompt: str, sys_prompt: str = None):
        if self.provider_ == "together":
            return self.llm_together_(prompt, sys_prompt)
        elif self.provider_ == "bedrock":
            return self.llm_bedrock_(prompt, sys_prompt)

    def llm_bedrock_(self, prompt: str, sys_prompt: str = None):
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.environ.get("BEDROCK_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("BEDROCK_SECRET_KEY"),
            region_name="us-west-2",
        )

        model_kwargs = {"temperature": 0.5, "top_p": 0.5, "max_gen_len": 512}
        model_id = {
            "llama31-405": "meta.llama3-1-405b-instruct-v1:0",
            "llama31-70": "meta.llama3-1-70b-instruct-v1:0",
            "llama31-8": "meta.llama3-1-8b-instruct-v1:0",
            "mixtral": "mistral.mixtral-8x7b-instruct-v0:1",
        }

        messages = []
        user_message = f"""<s>[INST]{prompt}[/INST]"""
        messages.append(
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        )
        modelid = model_id[self.model_]
        if self.model_ == "mixtral":

            response = bedrock_client.converse(
                modelId=modelid,
                messages=messages,
                inferenceConfig={"maxTokens": 800, "temperature": 0.7, "topP": 0.7},
                additionalModelRequestFields={"top_k": 50},
            )

        elif self.model_.startswith("llama"):

            response = bedrock_client.converse(
                modelId=modelid,
                messages=messages,
                system=[{"text": sys_prompt}],
                inferenceConfig={"maxTokens": 800, "temperature": 0.7, "topP": 0.7},
            )

        response_text = response["output"]["message"]["content"][0]["text"]

        tokens = {
            "prompt_tokens": response["usage"]["inputTokens"],
            "completion_tokens": response["usage"]["outputTokens"],
            "total_tokens": response["usage"]["totalTokens"],
        }

        return response_text, tokens

    def llm_together_(self, prompt: str, sys_prompt: str = None):
        client = openai.OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

        model_id = {
            "llama31-70": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "llama31-8": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "llama38": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            "llama38b": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }

        messages = []
        if sys_prompt is not None:
            messages.append({"role": "system", "content": sys_prompt})

        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model_id[self.model_], messages=messages
        )
        answer = response.choices[0].message.content
        tokens = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return answer, tokens


class JudgeLLM:
    def __init__(self, model="prometheus-eval/prometheus-7b-v2.0"):
        self.model = VLLM(model=model, dtype="bfloat16", max_model_len=4096)

    def generate(
        self,
        instruction: str,
        response: str,
        score_rubric: str,
        template: str,
        reference_answer: str = None,
    ):
        if template == "ref":
            judge = PrometheusEval(
                model=self.model, absolute_grade_template=ABSOLUTE_PROMPT
            )
            feedback, score = judge.single_absolute_grade(
                instruction=instruction,
                response=response,
                rubric=score_rubric,
                reference_answer=reference_answer,
            )
            return feedback, score
        elif template == "wo_ref":
            judge = PrometheusEval(
                model=self.model, absolute_grade_template=ABSOLUTE_PROMPT_WO_REF
            )
            feedback, score = judge.single_absolute_grade(
                instruction=instruction,
                response=response,
                rubric=score_rubric,
            )

        return feedback, score

    def generate_batch(
        self,
        instructions: List[str],
        responses: List[str],
        score_rubric: str,
        template: str,
        reference_answers: List[str] | None = None,
    ):
        if template == "ref":
            judge = PrometheusEval(
                model=self.model, absolute_grade_template=ABSOLUTE_PROMPT
            )
            feedback, score = judge.absolute_grade(
                instruction=instructions,
                response=responses,
                rubric=score_rubric,
                reference_answer=reference_answers,
            )
            return feedback, score
        elif template == "wo_ref":
            judge = PrometheusEval(
                model=self.model, absolute_grade_template=ABSOLUTE_PROMPT_WO_REF
            )
            feedback, score = judge.absolute_grade(
                instruction=instructions,
                response=responses,
                rubric=score_rubric,
            )

        return feedback, score
