import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

print("GROQ KEY FOUND:", os.getenv("GROQ_API_KEY") is not None)


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        # Load existing vector store
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()

        # Build if not found
        else:
            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        groq_api_key = os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(
            f"[INFO] Groq LLM initialized: {llm_model}"
        )

    def retrieve_chunks(
    self,
    query: str,
    top_k: int = 10
    ):
        return self.vectorstore.query(
            query,
            top_k=top_k
        )

    def search_and_summarize(
    self,
    query: str,
    top_k: int = 10
    ) -> str:

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        print("\n========== RETRIEVED CONTEXT ==========\n")
        print(context[:5000])
        print("\n=======================================\n")

        if not context:
            return "No relevant documents found."

        prompt = f"""
                You are an expert AI assistant specialized in answering questions from uploaded documents.

                Instructions:
                - Answer ONLY from the provided context.
                - Give detailed explanations.
                - Use multiple paragraphs when needed.
                - Use bullet points when appropriate.
                - Include examples from the document whenever available.
                - Do not make up information.
                - If the answer is not found in the context, say:
                "I could not find this information in the uploaded documents."

                Question:
                {query}

                Context:
                {context}

                Detailed Answer:
                """

        response = self.llm.invoke(prompt)

        return response.content











