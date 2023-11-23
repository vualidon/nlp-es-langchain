import openai

openai.api_key = "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"

def generate(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """"""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9
    )

    return response['choices'][0]['message']['content']