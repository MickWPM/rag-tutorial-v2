import argparse
import os
import shutil
#from langchain.document_loaders.pdf import PyPDFDirectoryLoader #not used

from langchain_community.document_loaders import DirectoryLoader #from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader #from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma #from langchain.vectorstores.chroma import Chroma

import time
from datetime import datetime

CHROMA_PATH = "chroma"
DATA_PATH = "eq_data"


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    print(f'{get_timestamp()} Documents loaded')

    print(f'{get_timestamp()} Splitting documents...')
    chunks = split_documents(documents)
    print(f'{get_timestamp()} Documents split')

    add_to_chroma(chunks)
    print(f'{get_timestamp()} Complete')


def load_documents():
    print(f'{get_timestamp()} Loading documents from {DATA_PATH}...')
    #document_loader = TextLoader(DATA_PATH)
    text_loader_kwargs={'autodetect_encoding': True}
    document_loader = DirectoryLoader(DATA_PATH, glob='**/*.txt', loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    print(f'{get_timestamp()}Document loaded returned')
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], batch_size: int = 160):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(),
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Calculate Page IDs.
    print(f'{get_timestamp()} Calculating chunk IDs')
    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f'{get_timestamp()} Calculated chunk IDs')
    
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"{get_timestamp()}Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"{get_timestamp()}👉 Adding new documents: {len(new_chunks)}")

        # Adding in batches for feedback
        total_added = 0
        start_time = time.time()  # Record the starting time

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            new_chunk_ids = [chunk.metadata["id"] for chunk in batch]
            
            # Record the time at the start of the batch
            batch_start_time = time.time()

            db.add_documents(batch, ids=new_chunk_ids)
            db.persist()

            # Record the time after the batch is processed
            batch_end_time = time.time()
            batch_time_taken = batch_end_time - batch_start_time  # Time taken for this batch

            total_added += len(batch)
            
            # Estimate remaining time (based on the average time per batch)
            batches_remaining = len(new_chunks) - total_added
            average_batch_time = (batch_end_time - start_time) / total_added if total_added > 0 else 0
            estimated_time_remaining = average_batch_time * batches_remaining
            
            # Format the estimated time remaining in h:mm:ss
            formatted_estimated_time = format_time(estimated_time_remaining)

            # Calculate the estimated finish time (current time + estimated time remaining)
            estimated_finish_timestamp = time.time() + estimated_time_remaining
            formatted_estimated_finish_time = format_time(estimated_finish_timestamp - time.time())


            # Print progress with timestamp, estimated time remaining, and estimated finish time
            print(f"{get_timestamp()} - 🔄 Added {total_added}/{len(new_chunks)} documents. "
                  f"Batch time: {batch_time_taken:.2f}s. "
                  f"Estimated time remaining: {formatted_estimated_time}. "
                  f"Estimated finish time: {formatted_estimated_finish_time}")

        print(f"{get_timestamp()} - ✅ All new documents added.")
    else:
        print(f"{get_timestamp()} - ✅ No new documents to add")


def format_time(seconds: float) -> str:
    """Convert seconds to h:mm:ss format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_timestamp() -> str:
    """Return the current timestamp in a readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")

def calculate_chunk_ids(chunks):
    print(f'{get_timestamp()} - 🔄 Calculating chunk IDs...')

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
