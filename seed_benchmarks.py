from creds import openai_key
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from faker import Faker
import random

llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)

def build_chain(prompt_template):
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain

def run_seeds(llm_chain, seeds):
    output_list = []
    for seed in seeds:
        output_list.append(llm_chain.run(seed=seed))
    return output_list

number_seeds = range(10)

big_number_seeds = [random.randint(1000000000, 9999999999) for _ in range(10)]

faker = Faker()
word_seeds = [faker.word() for _ in range(10)]

mood_prompt = PromptTemplate(
    input_variables=["seed"],
    template="""
    Mood: {seed}

    Give me one baby name
    """,
)

mood_chain = build_chain(mood_prompt)

set_seed_prompt = PromptTemplate(
    input_variables=["seed"],
    template="""
    Set seed: {seed}

    Give me one baby name
    """,
)

set_seed_chain = build_chain(set_seed_prompt)

inspired_prompt = PromptTemplate(
    input_variables=["seed"],
    template="""
    Be inspired by this word: {seed}

    Give me one baby name
    """,
)

inspired_chain = build_chain(inspired_prompt)

raw_seed_prompt = PromptTemplate(
    input_variables=["seed"],
    template="""
    {seed}

    Give me one baby name
    """,
)

raw_seed_chain = build_chain(raw_seed_prompt)
