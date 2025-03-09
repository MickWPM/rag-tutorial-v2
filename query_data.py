import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

#OLLAMA_MODEL = "llama3.2"
OLLAMA_MODEL = "mistral"

PROMPT_TEMPLATE = """
You will recieve a block of text with an item from a video game and a question about the item. The data format in the text file is plain english but some of the formatting clarity is lost in the translation from web interface to text. A few format notes:
Drops From: underneath this section there will either be text stating it doesnt drop or a list where the Zone is left justafied and the monsters that drop the item in that zone are indented by a space. The next zone list starts with the zone name and no indent.
Sold By: 
This data is from a table. The first part of this section states Zone, merchant name, area and loc. Not all of these may contain text. As an example:

 Kael Drakkel



 Kellek Felhammer



  

 (267, 1867)

In this case, the zone is Kael Drakkel, the merchant is Kellek Felhammer and his location is (267, 1867)

Keep this context in mind when answering questions

Questions will be provided with the context for the item in question.  Only answer the question asked unless specifically prompted for more.
ONLY use the context provided in the question in the answer.

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"})

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    
    print ("DB RESULTS:")
    for result in results:
        for doc, score in results:
            print(f"id: {doc.metadata['id']}. score: {score}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model=OLLAMA_MODEL)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
