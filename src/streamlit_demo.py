from response import generate
from search import bm25_search, semantic_search
import streamlit as st


st.title("# DEMO")
query = st.text_input("Input your question here")
btn_submit = st.button("SUBMIT")

references = semantic_search(query, 5)
references = [reference['_source']['content'] for reference in references]
references = '\n\n'.join(references)

if btn_submit:
    answer = generate(f"""
    Từ thông tin: ```{references}```

    Hãy trả lời câu hỏi: ```{query}```
    """)

    st.write(answer)