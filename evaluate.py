from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from ragas.metrics import AspectCritic
from ragas import SingleTurnSample, evaluate
from ragas.metrics import faithfulness, context_precision, answer_correctness
from datasets import Dataset 

from dotenv import load_dotenv
import os

load_dotenv()
apikey = os.environ.get("OPENAI_API_KEY")

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=apikey))

async def evaluateResponse(user_input, context, llm_response, reference):
    sample = SingleTurnSample(
        user_input = user_input,
        retrieved_contexts = context,
        response = llm_response,
        reference = reference,
    )
    dataset = Dataset.from_list([sample.model_dump()])
    results = evaluate(dataset=dataset, metrics=[faithfulness,context_precision,answer_correctness])
    dataframe = results.to_pandas()
    print("Evaluation Scores: ")
    print(dataframe)

    scorer = AspectCritic(name="correctness",definition="Does the answer accurately reflect the given financial information?")
    scorer.llm = evaluator_llm
    ascore = await scorer.single_turn_ascore(sample)
    print("Single Turn A-Score: ", ascore)