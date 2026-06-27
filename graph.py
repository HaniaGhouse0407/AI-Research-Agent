"""
LangGraph Research Agent: multi-agent pipeline for automated research synthesis.
Implements: Query Decomposition → Parallel Search → Synthesis → Critique → Report
"""
from __future__ import annotations
from typing import TypedDict, List, Dict, Optional, Annotated
import operator, logging

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """Shared state passed between graph nodes."""
    original_query: str
    sub_queries: List[str]
    search_results: Annotated[List[Dict], operator.add]  # accumulate across parallel nodes
    synthesized_draft: str
    critique: str
    final_report: str
    iteration: int
    max_iterations: int


def build_research_graph(llm=None, tools=None):
    """
    Build and compile the LangGraph research agent.

    Graph structure:
        decompose → [arxiv_search, semantic_search] → synthesize → critique
                                                            ↑___________| (loop)
        → finalize
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("pip install langgraph langchain-core")

    from .nodes import (
        decompose_query, search_arxiv, search_semantic_scholar,
        synthesize_results, critique_draft, finalize_report
    )

    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("decompose", decompose_query)
    graph.add_node("arxiv_search", search_arxiv)
    graph.add_node("semantic_search", search_semantic_scholar)
    graph.add_node("synthesize", synthesize_results)
    graph.add_node("critique", critique_draft)
    graph.add_node("finalize", finalize_report)

    # Entry
    graph.set_entry_point("decompose")

    # Decompose → parallel search
    graph.add_edge("decompose", "arxiv_search")
    graph.add_edge("decompose", "semantic_search")

    # Searches → synthesize
    graph.add_edge("arxiv_search", "synthesize")
    graph.add_edge("semantic_search", "synthesize")

    # Synthesize → critique
    graph.add_edge("synthesize", "critique")

    # Critique → loop or finalize
    def should_continue(state: ResearchState):
        if state["iteration"] >= state["max_iterations"]:
            return "finalize"
        if _quality_ok(state["critique"]):
            return "finalize"
        return "synthesize"   # re-synthesize with critique feedback

    graph.add_conditional_edges("critique", should_continue, {
        "synthesize": "synthesize",
        "finalize": "finalize",
    })
    graph.add_edge("finalize", END)

    return graph.compile()


def _quality_ok(critique: str) -> bool:
    """Heuristic: check if critique says the report is good enough."""
    approve_phrases = ["sufficient", "comprehensive", "well-cited", "complete", "adequate"]
    return any(phrase in critique.lower() for phrase in approve_phrases)


class ResearchPipeline:
    """High-level interface for running the research agent."""

    def __init__(self, max_iterations: int = 2, llm=None):
        self.max_iterations = max_iterations
        self.llm = llm or self._default_llm()
        self._graph = None

    def _default_llm(self):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except ImportError:
            return None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = build_research_graph(llm=self.llm)
        return self._graph

    def run(self, query: str) -> Dict:
        """Run the full research pipeline."""
        initial_state = ResearchState(
            original_query=query,
            sub_queries=[],
            search_results=[],
            synthesized_draft="",
            critique="",
            final_report="",
            iteration=0,
            max_iterations=self.max_iterations,
        )
        logger.info(f"Starting research: {query[:80]}...")
        result = self.graph.invoke(initial_state)
        return {
            "query": query,
            "report": result["final_report"],
            "sources": [r.get("url", "") for r in result["search_results"]],
            "iterations": result["iteration"],
        }
