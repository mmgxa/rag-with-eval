import boto3
import json
import os
from dotenv import load_dotenv
from .config import get_project_root
from fastembed import TextEmbedding
from pydantic import BaseModel
from typing import Literal

load_dotenv(get_project_root() + "/.env")


class Providers(BaseModel):
    provider: Literal["bedrock", "fastembed"]


class Embedder:
    def __init__(self, model, provider: str):

        self.provider_ = Providers(provider=provider).provider

        self.model_ = (
            TextEmbedding(model_name=model) if self.provider_ == "fastembed" else model
        )
        self.ndims = len(self.embed("foo"))

    def embed(self, input_text: str):
        if self.provider_ == "fastembed":
            return self.embed_fastembed_(input_text)
        elif self.provider_ == "bedrock":
            return self.embed_bedrock_(input_text)

    def embed_bedrock_(self, input_text: str):
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.environ.get("BEDROCK_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("BEDROCK_SECRET_KEY"),
            region_name="us-west-2",
        )
        native_request = {
            "inputText": input_text,
            "dimensions": 256,
            "normalize": False,
        }
        request = json.dumps(native_request)
        response = bedrock_client.invoke_model(modelId=self.model_, body=request)
        model_response = json.loads(response["body"].read())
        embedding = model_response["embedding"]
        return embedding

    def embed_fastembed_(self, input_text: str):
        embedding = list(self.model_.embed(input_text))[0].tolist()
        return embedding
