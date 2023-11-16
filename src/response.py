import openai

openai.api_key = "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"

def generate(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": """Bạn là một người hướng dẫn trong trường học.\
             Công việc của bạn là giúp cho người khác hiểu được thông tin của trường một cách chi tiết nhất."""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )

    return response['choices'][0]['message']['content']