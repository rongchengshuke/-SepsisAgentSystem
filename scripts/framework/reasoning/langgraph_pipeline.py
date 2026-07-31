from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

from scripts.framework.reasoning.retriever import TfidfKnowledgeBase
from scripts.framework.reasoning.sample_context import format_patient_summary
from scripts.framework.reasoning.prompts import (
    CLINICAL_REASONING_SYSTEM_PROMPT,
    build_clinical_reasoning_user_prompt,
)

try:
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
except ImportError:
    HumanMessage = None
    ChatPromptTemplate = None
    ChatOpenAI = None
    StateGraph = None
    START = None
    END = None


SYSTEM_PROMPT = CLINICAL_REASONING_SYSTEM_PROMPT


OPENAI_COMPATIBLE_BACKENDS = {"deepseek", "glm", "qwen"}


DEFAULT_SYNTHETIC_SUMMARY: dict[str, object] = {
    "sample_id": "synthetic_elderly_sepsis_001",
    "window_start": 0,
    "window_end": 24,
    "age": "78",
    "gender": "female",
    "bmi": "26.40",
    "cci_score": "6.00",
    "gnri": "102.10",
    "hr": "103.00",
    "sbp": "101.00",
    "mbp": "81.00",
    "resp_rate": "19.00",
    "temperature": "35.60",
    "spo2": "100.00",
    "urineoutput": "260.00",
    "gcs": "14.00",
    "sofa": "4.00",
    "lactate": "2.00",
    "creatinine": "1.20",
    "bilirubin": "0.30",
    "platelet": "357.00",
    "wbc": "24.20",
    "ph": "7.36",
    "pao2fio2": "280.00",
    "inr": "1.40",
    "vasopressors": "0.00",
    "ventilation": "1.00",
}


class ReasoningState(TypedDict, total=False):
    summary: dict[str, object]
    query: str
    top_k: int
    retrieved_docs: List[Dict[str, Union[str, float]]]
    rag_enabled: bool
    patient_summary_text: str
    knowledge_text: str
    note_text: str
    image_data_url: str
    image_path: str
    reasoning_text: str
    backend: str


def build_synthetic_case() -> dict[str, object]:
    summary = dict(DEFAULT_SYNTHETIC_SUMMARY)
    query = (
        f"elderly sepsis prognosis age {summary['age']} cci {summary['cci_score']} sofa {summary['sofa']} "
        f"lactate {summary['lactate']} creatinine {summary['creatinine']} bilirubin {summary['bilirubin']} "
        f"platelet {summary['platelet']} wbc {summary['wbc']} mbp {summary['mbp']} spo2 {summary['spo2']} "
        f"vasopressors {summary['vasopressors']} ventilation {summary['ventilation']}"
    )
    return {"summary": summary, "query": query}


def render_knowledge(results: List[Dict[str, Union[str, float]]]) -> str:
    blocks = []
    for idx, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Knowledge {idx}] {result['title']}",
                    f"Source: {result['source']}",
                    str(result["text"]),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_mock_reasoning(summary: dict[str, object], retrieved_docs: List[Dict[str, Union[str, float]]]) -> str:
    return json.dumps(
        {
            "structured_summary": {
                "vital_signs": f"Current hemodynamics include MBP {summary['mbp']} and vasopressor exposure {summary['vasopressors']}. Respiratory support includes SpO2 {summary['spo2']} and ventilation {summary['ventilation']}.",
                "laboratory_findings": f"Observed laboratory status includes lactate {summary['lactate']}, creatinine {summary['creatinine']}, bilirubin {summary['bilirubin']}, platelet {summary['platelet']}, and INR {summary['inr']}.",
                "therapeutic_interventions": f"Treatment status includes vasopressors {summary['vasopressors']} and mechanical ventilation {summary['ventilation']}.",
            },
            "text_cleaned": "unavailable",
            "reasoning_dimensions": {
                "hemodynamic_stability": f"MBP {summary['mbp']} with vasopressor status {summary['vasopressors']}.",
                "respiratory_oxygenation_burden": f"SpO2 {summary['spo2']} with ventilation status {summary['ventilation']}.",
                "infection_immune_stress": f"WBC {summary['wbc']} with lactate {summary['lactate']}.",
                "coagulation_microcirculation": f"Platelet {summary['platelet']} and INR {summary['inr']}.",
                "renal_internal_environment": f"Creatinine {summary['creatinine']}.",
                "metabolic_perfusion_acid_base": f"Lactate {summary['lactate']}.",
                "neurological_consciousness_status": f"GCS {summary['gcs']}.",
                "nutrition_frailty_burden": f"Age {summary['age']}, CCI {summary['cci_score']}, GNRI {summary['gnri']}.",
            },
            "overall_assessment": "Objective current-state summary generated in mock mode.",
        },
        ensure_ascii=False,
    )


class LangGraphReasoner:
    def __init__(
        self,
        docs_dir: Union[str, Path, list[str], list[Path]],
        api_key: Optional[str],
        model_name: str,
        base_url: str,
        backend: str = "deepseek",
        temperature: float = 0.0,
        enable_rag: bool = False,
    ):
        self.backend = backend
        self.retriever = TfidfKnowledgeBase(docs_dir)
        self.enable_rag = enable_rag
        self.prompt = None
        self.llm = None

        if self.backend in OPENAI_COMPATIBLE_BACKENDS:
            self._validate_langgraph_dependencies()
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    (
                        "human",
                        "{user_prompt}",
                    ),
                ]
            )
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_retries=2,
            )

        if StateGraph is None or START is None or END is None:
            raise ImportError("langgraph is not installed. Install it in the target environment first.")

        graph = StateGraph(ReasoningState)
        graph.add_node("retrieve_knowledge", self.retrieve_knowledge)
        graph.add_node("build_prompt", self.build_prompt)
        graph.add_node("generate_reasoning", self.generate_reasoning)
        graph.add_edge(START, "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "build_prompt")
        graph.add_edge("build_prompt", "generate_reasoning")
        graph.add_edge("generate_reasoning", END)
        self.graph = graph.compile()

    @staticmethod
    def _validate_langgraph_dependencies() -> None:
        if ChatPromptTemplate is None or ChatOpenAI is None:
            raise ImportError(
                "langchain_openai and langchain_core are required for OpenAI-compatible reasoning mode. "
                "Install them in the target environment first."
            )

    def retrieve_knowledge(self, state: ReasoningState) -> dict[str, Any]:
        if not self.enable_rag:
            return {"retrieved_docs": [], "rag_enabled": False}
        retrieved_docs = self.retriever.search(state["query"], top_k=state.get("top_k", 3))
        return {"retrieved_docs": retrieved_docs, "rag_enabled": True}

    def build_prompt(self, state: ReasoningState) -> dict[str, Any]:
        patient_summary_text = format_patient_summary(state["summary"])
        knowledge_text = (
            render_knowledge(state.get("retrieved_docs", []))
            if state.get("rag_enabled")
            else "RAG disabled for this experiment."
        )
        return {
            "patient_summary_text": patient_summary_text,
            "knowledge_text": knowledge_text,
            "note_text": str(state.get("note_text", "")).strip(),
            "image_data_url": str(state.get("image_data_url", "")).strip(),
            "image_path": str(state.get("image_path", "")).strip(),
        }

    def generate_reasoning(self, state: ReasoningState) -> dict[str, Any]:
        if self.backend == "mock":
            reasoning_text = build_mock_reasoning(state["summary"], state.get("retrieved_docs", []))
        else:
            assert self.prompt is not None
            assert self.llm is not None
            note_text = str(state.get("note_text", "")).strip()
            image_data_url = str(state.get("image_data_url", "")).strip()
            image_path = str(state.get("image_path", "")).strip()
            if note_text:
                note_text = note_text[:6000]

            if image_data_url and HumanMessage is not None:
                user_prompt = build_clinical_reasoning_user_prompt(state["summary"], note_text)
                text_block = "\n\n".join(
                    [
                        SYSTEM_PROMPT,
                        user_prompt,
                        (
                            f"Attached image path: {image_path}\n"
                            "The image is a placeholder white-background image because no aligned image modality is available in the local dataset sample."
                        ),
                    ]
                )
                try:
                    response = self.llm.invoke(
                        [
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": text_block},
                                    {"type": "image_url", "image_url": {"url": image_data_url}},
                                ]
                            )
                        ]
                    )
                except Exception:
                    user_prompt = build_clinical_reasoning_user_prompt(state["summary"], note_text)
                    messages = self.prompt.format_messages(
                        user_prompt=user_prompt,
                    )
                    response = self.llm.invoke(messages)
            else:
                user_prompt = build_clinical_reasoning_user_prompt(state["summary"], note_text)
                messages = self.prompt.format_messages(
                    user_prompt=user_prompt,
                )
                response = self.llm.invoke(messages)
            reasoning_text = self._extract_content(response.content)
        return {"reasoning_text": reasoning_text}

    @staticmethod
    def _extract_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
            return "\n".join(part.strip() for part in parts if part.strip()).strip()
        return str(content).strip()

    def invoke(
        self,
        case: dict[str, object],
        top_k: int = 3,
        note_text: str = "",
        image_data_url: str = "",
        image_path: str = "",
    ) -> dict[str, Any]:
        state: ReasoningState = {
            "summary": dict(case["summary"]),
            "query": str(case["query"]),
            "top_k": top_k,
            "backend": self.backend,
            "note_text": note_text,
            "image_data_url": image_data_url,
            "image_path": image_path,
        }
        return self.graph.invoke(state)
