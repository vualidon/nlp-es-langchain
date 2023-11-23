import PyPDF2

def read_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page in range(num_pages):
        page_obj = pdf_reader.pages[page]
        text += page_obj.extract_text()
    pdf_file_obj.close()
    return text

def split_document(data_path):
    document = read_pdf(data_path)
    articles = document.split('Điều')[1:]
    articles = ['Điều' + statute for statute in articles]
    return articles
    
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    # docs = text_splitter.split_documents(document)
    # return docs

# # Use the function
# text = read_pdf('./law_data/1.pdf')
# print(text)