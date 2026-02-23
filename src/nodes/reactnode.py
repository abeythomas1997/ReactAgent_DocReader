"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
import builtins as _builtins
import uuid as _uuid

# Ensure 'uuid' is available globally for libraries that reference it in type hints
setattr(_builtins, "uuid", _uuid)
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from pydantic import BaseModel, Field

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        # Force explicit JSON schemas: {"query": "<string>"} for both tools.
        # This avoids malformed tool calls and matches what the Groq model is emitting.
        class RetrieverArgs(BaseModel):
            query: str = Field(..., description="Search query for the RAG retriever")

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = StructuredTool.from_function(
            name="retriever",
            description="Fetch passages from the indexed document corpus. Input must be JSON with a 'query' field.",
            func=retriever_tool_fn,
            args_schema=RetrieverArgs,
        )

        class WikipediaArgs(BaseModel):
            query: str = Field(..., description="Wikipedia search query")

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        def wikipedia_tool_fn(query: str) -> str:
            return wiki.run(query)

        wikipedia_tool = StructuredTool.from_function(
            name="wikipedia",
            description="Search Wikipedia for general knowledge. Input must be JSON with a 'query' field.",
            func=wikipedia_tool_fn,
            args_schema=WikipediaArgs,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer the 'retriever' tool for questions about the indexed documents; "
            "use 'wikipedia' only for general background knowledge when needed. "
            "When calling a tool, you MUST pass valid JSON arguments. "
            "For wikipedia, the arguments must look like: {\"query\": \"...\"}. "
            "Return only the final useful answer."
        )
        self._agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )
