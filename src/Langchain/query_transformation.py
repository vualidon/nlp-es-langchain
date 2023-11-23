import os
import json

import dotenv
dotenv = dotenv.load_dotenv()

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

output_parser = LineListOutputParser()

llm = ChatOpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)


if __name__=="__main__":
    # query = "Trường đại học Tôn Đức Thắng có bao nhiêu cơ sở"

    query = "Trường đại học Công nghệ thông tin xét tuyển các hình thức nào?"

    

    # Chain
    

    print(llm_chain.run({"question": query}))
    # print(gen_queries(query))