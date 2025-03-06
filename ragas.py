from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from ragas.metrics import AspectCritic
from ragas import SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, ContextPrecision
from datasets import Dataset 


evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))


def evaluateResponse(user_input, context, llm_response, reference):
    sample = SingleTurnSample(
        user_input = user_input,
        retrieved_contexts = context,
        response = llm_response,
        reference = reference,
    )
    dataset = Dataset(sample.model_dump())
    results = evaluate(dataset=dataset, metrics=[Faithfulness,FactualCorrectness,ContextPrecision])
    dataframe = results.to_pandas()
    print("Evaluation Scores: ")
    print(dataframe)

    scorer = AspectCritic(name="correctness",definition="Does the answer accurately reflect the given financial information?")
    scorer.llm = evaluator_llm
    ascore = scorer.single_turn_ascore(sample)
    print("Single Turn A-Score: " + ascore)