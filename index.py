import streamlit as st
import os
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage

document_store = InMemoryDocumentStore()
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
writer = DocumentWriter(document_store = document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=fetcher, name="fetcher")
indexing_pipeline.add_component(instance=converter, name="converter")
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=20))
indexing_pipeline.add_component(instance=writer, name="writer")
indexing_pipeline.connect("fetcher.streams", "converter.sources")
indexing_pipeline.connect("converter.documents", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "writer.documents")

os.environ["OPENAI_API_KEY"] = st.secrets['open_ai_api_key']
prompt_template = """
    Donot make anything up. Only speak from what you know or the provided documents.
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    \nQuestion: {{question}}
    \nAnswer:
    """

def create_rag_pipeline(placeholder):
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := placeholder.chat_input("What is up?"):
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(ChatMessage.from_user(prompt))
        res = rag_pipeline.run(
                {
                    "retriever": {"query": prompt},
                    "prompt_builder": {"question": prompt},
                    "answer_builder": {"query": prompt},
                }
            )

        print(res)
        st.session_state.messages.append(ChatMessage.from_assistant(res['answer_builder']['answers'][0]))
        st.chat_message("assistant").markdown(res['answer_builder']['answers'][0].data)
    
placeholder = st.empty()
if website_url := placeholder.text_input('Website URL'):
    indexing_pipeline.run(data={"fetcher": {"urls": [website_url]}})
    create_rag_pipeline(placeholder)



