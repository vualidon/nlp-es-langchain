import dotenv
dotenv = dotenv.load_dotenv()
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_questions(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines."""},
            {"role": "user", "content": f"Original question: {question}"}
        ],
        temperature=0
    )

    questions = response['choices'][0]['message']['content'].split("\n")

    return questions


def generate_answer(doc, question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": """You are an AI language model assistant. 
             Given the text below, answer the user's question as thoroughly and informatively as possible. 
             Provide explanations and examples to enhance understanding."""},
            {"role": "user", "content": f"""Document: ```{doc}``` 
             Question: {question}.
             Answer: """}
        ],
        temperature=0
    )

    questions = response['choices'][0]['message']['content']

    return questions