from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import LiteLLMModel #DeepEval allows you to use any model supported by LiteLLM to run evals
#from evals.harness import load_goldens, summarize_by_metric, print_summary
from dotenv import load_dotenv
import json
import os
from src.retriever import build_retriever
import warnings

# Prevent LiteLLM from passing unsupported arguments to OpenRouter
os.environ["LITELLM_DROP_PARAMS"] = "True"

# Read the .env file and place its contents into the system environment
load_dotenv() 

# Retrieve the open_router_api_key variable using the standard os library
hf_api_key = os.getenv("HF_TOKEN")

# 1. Initialize OpenAI via LiteLLM (Prefix with "openai/")
# Use "mistral/mistral-small" "
# openai_judge = OpenRouterModel(
#     model="nvidia/nemotron-3-ultra-550b-a55b:free",
#     api_key=open_router_api_key,
#     base_url="https://openrouter.ai/api/v1",
#     temperature=0
# )
# 1. Suppress fallback warnings
warnings.filterwarnings("ignore", category=UserWarning, module="deepeval")

model = LiteLLMModel(
    # Free serverless tier endpoint for Qwen 2.5 0.5B
    model="huggingface/Qwen/Qwen2.5-0.5B-Instruct", 
    api_key=hf_api_key,
    #base_url="https://huggingface.co",
    temperature=0
)

GOLDEN_PATH = r"golden_dataset\retriever_golden_ds.json"
JUDGE_MODEL = model  # Use the initialized LiteLLMModel for evaluation
THRESHOLD = 0.7

#Loading the golden data for evaluation
with open(GOLDEN_PATH) as f:
    golden_data = json.load(f)

retriever = build_retriever()

test_cases = []
for golden in golden_data:
    retrieved = retriever.invoke(golden["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=golden["query"],
            expected_output=golden["expected_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )

metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]


evaluate(
        test_cases=test_cases,
        metrics=metrics,
        # hyperparameters={
        #     "retriever": "reranker",          # vs "reranked" when you swap it in
        #     "embedding_model": "text-embedding-3-large",
        #     "chunk_size": 1000,
        #     "chunk_overlap": 150,
        #     "top_k": 3,
        #     "judge_model": JUDGE_MODEL,
        #     "golden_set": GOLDEN_PATH,
        # },
    )