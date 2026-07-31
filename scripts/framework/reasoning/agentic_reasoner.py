from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from scripts.framework.reasoning.experience_store import ExperienceStore
from scripts.framework.reasoning.langgraph_pipeline import (
    DEFAULT_SYNTHETIC_SUMMARY,
    LangGraphReasoner,
    build_mock_reasoning,
    render_knowledge,
)
from scripts.framework.reasoning.prompts import (
    CLINICAL_REASONING_SYSTEM_PROMPT,
    build_focused_clinical_reasoning_user_prompt,
)
from scripts.framework.reasoning.retriever import TfidfKnowledgeBase
from scripts.framework.reasoning.sample_context import format_patient_summary

try:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
except ImportError:
    HumanMessage = None
    ChatOpenAI = None
    StateGraph = None
    START = None
    END = None


DOCTOR_SPECS = [
    {
        "doctor_id": "doctor_1",
        "title": "Hemodynamic and Respiratory Doctor",
        "focus": "Focus on perfusion, blood pressure trends, vasopressors, oxygenation, and ventilation support.",
    },
    {
        "doctor_id": "doctor_2",
        "title": "Organ Dysfunction and Laboratory Doctor",
        "focus": "Focus on SOFA, lactate, creatinine, BNP, bilirubin, platelet, INR, pH, infection burden, and multi-organ failure.",
    },
    {
        "doctor_id": "doctor_3",
        "title": "History, Treatment, and Global Status Doctor",
        "focus": "Focus on age, comorbidity, nutrition, treatment intensity, admission history, and baseline reserve.",
    },
]


COLACARE_SYSTEM_PROMPT = CLINICAL_REASONING_SYSTEM_PROMPT


META_SYSTEM_PROMPT = CLINICAL_REASONING_SYSTEM_PROMPT


EVOLUTION_SYSTEM_PROMPT = """You are maintaining an experience base for a clinical reasoning agent.
Distill reusable strategic principles from the completed consultation. Principles should be short, generalizable, and clinically actionable."""


OPENAI_COMPATIBLE_BACKENDS = {"deepseek", "glm", "qwen"}
MEMORY_UPDATE_POLICIES = {"all", "error_only", "uncertain_or_error"}


DOCTOR_PROMPT_SCOPES = {
    "doctor_1": {
        "static_keys": ["age", "gender"],
        "dynamic_keys": ["heart_rate", "blood_pressure", "spo2", "vasopressors", "ventilation"],
    },
    "doctor_2": {
        "static_keys": ["age", "gender", "cci_score"],
        "dynamic_keys": ["glucose", "creatinine", "bnp", "platelet", "neutrophils", "lymphocytes"],
    },
    "doctor_3": {
        "static_keys": ["age", "gender", "bmi", "cci_score", "gnri"],
        "dynamic_keys": ["vasopressors", "ventilation"],
    },
}


class AgenticReasoningState(TypedDict, total=False):
    summary: dict[str, object]
    query: str
    top_k: int
    note_text: str
    image_data_url: str
    image_path: str
    patient_summary_text: str
    knowledge_text: str
    retrieved_docs: list[dict[str, Any]]
    rag_enabled: bool
    experience_hits: list[dict[str, Any]]
    experience_text: str
    doctor_experience_hits: dict[str, list[dict[str, Any]]]
    doctor_experience_text: dict[str, str]
    doctor_reviews: list[dict[str, Any]]
    meta_system_prompt: str
    meta_user_prompt: str
    meta_report: str
    current_round: int
    consultation_feedback: list[dict[str, str]]
    continue_discussion: bool
    final_report: str
    reasoning_text: str
    evolution_result: dict[str, Any]


def build_synthetic_case() -> dict[str, object]:
    summary = dict(DEFAULT_SYNTHETIC_SUMMARY)
    query = (
        f"elderly sepsis prognosis age {summary['age']} cci {summary['cci_score']} sofa {summary['sofa']} "
        f"lactate {summary['lactate']} creatinine {summary['creatinine']} bilirubin {summary['bilirubin']} "
        f"platelet {summary['platelet']} wbc {summary['wbc']} mbp {summary['mbp']} spo2 {summary['spo2']} "
        f"vasopressors {summary['vasopressors']} ventilation {summary['ventilation']}"
    )
    return {"summary": summary, "query": query}


def _safe_json_loads(text: str) -> Any:
    cleaned = text.strip()
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    return json.loads(cleaned)


def _json_object_or_fallback(value: Any, fallback: dict[str, str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        rendered = "; ".join(
            json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
            for item in value
        )
        merged = dict(fallback)
        merged["comment"] = rendered
        merged["evidence"] = rendered
        return merged
    merged = dict(fallback)
    merged["comment"] = str(value)
    merged["evidence"] = str(value)
    return merged


def _stringify_experience_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "No prior experience principles retrieved."
    blocks = []
    for idx, hit in enumerate(hits, start=1):
        triples = hit.get("triples", [])
        triple_text = "; ".join(
            f"{item.get('head', '')} | {item.get('relation', '')} | {item.get('tail', '')}"
            for item in triples
            if isinstance(item, dict)
        )
        blocks.append(
            "\n".join(
                [
                    f"[Principle {idx}] {hit.get('description', '')}",
                    f"Category: {hit.get('category', 'guiding')}",
                    f"Historical score: {float(hit.get('score', 0.5)):.2f}",
                    f"Triples: {triple_text or 'none'}",
                ]
            )
        )
    return "\n\n".join(blocks)


class ColaCareEvolveReasoner:
    def __init__(
        self,
        docs_dir: str | Path | list[str] | list[Path],
        api_key: str | None,
        model_name: str,
        base_url: str,
        backend: str = "deepseek",
        temperature: float = 0.0,
        experience_store_path: str | Path = "data/reasoning/colacare_evolver_experience.json",
        experience_store_paths: dict[str, str | Path] | None = None,
        experience_top_k: int = 3,
        max_consult_rounds: int = 2,
        enable_self_evolution: bool = True,
        self_evolution_mode: str = "inline",
        memory_update_policy: str = "all",
        enable_rag: bool = False,
    ):
        self.backend = backend
        self.retriever = TfidfKnowledgeBase(docs_dir)
        self.enable_rag = enable_rag
        self.experience_top_k = experience_top_k
        self.max_consult_rounds = max(1, max_consult_rounds)
        self.enable_self_evolution = enable_self_evolution
        self.self_evolution_mode = str(self_evolution_mode).strip().lower()
        if self.self_evolution_mode not in {"inline", "post_prediction"}:
            self.self_evolution_mode = "inline"
        self.memory_update_policy = self._normalize_memory_update_policy(memory_update_policy)
        self.chat_max_attempts = max(1, int(os.environ.get("AGENTIC_CHAT_MAX_ATTEMPTS", "4")))
        self.chat_retry_wait_seconds = max(0.0, float(os.environ.get("AGENTIC_CHAT_RETRY_WAIT_SECONDS", "45")))
        self.chat_throttle_seconds = max(0.0, float(os.environ.get("AGENTIC_CHAT_THROTTLE_SECONDS", "0")))
        self.chat_timeout_seconds = max(1.0, float(os.environ.get("AGENTIC_CHAT_TIMEOUT_SECONDS", "90")))
        self.split_experience_stores = bool(experience_store_paths)
        self.shared_experience_store: ExperienceStore | None = None
        self.doctor_experience_stores: dict[str, ExperienceStore] = {}
        if self.split_experience_stores:
            normalized_paths = {str(key): Path(value) for key, value in (experience_store_paths or {}).items()}
            meta_path = normalized_paths.get("meta_agent") or normalized_paths.get("meta") or Path(
                experience_store_path
            )
            self.meta_experience_store = ExperienceStore(meta_path)
            for spec in DOCTOR_SPECS:
                doctor_id = spec["doctor_id"]
                doctor_path = normalized_paths.get(doctor_id)
                if doctor_path is None:
                    doctor_path = meta_path.parent / f"{doctor_id}_experience.json"
                self.doctor_experience_stores[doctor_id] = ExperienceStore(doctor_path)
        else:
            shared_store = ExperienceStore(experience_store_path)
            self.shared_experience_store = shared_store
            self.meta_experience_store = shared_store
            for spec in DOCTOR_SPECS:
                self.doctor_experience_stores[spec["doctor_id"]] = shared_store
        self.experience_store = self.meta_experience_store
        self.llm = None
        if self.backend in OPENAI_COMPATIBLE_BACKENDS:
            if ChatOpenAI is None:
                raise ImportError("langchain_openai is required for OpenAI-compatible reasoning mode.")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_retries=2,
                timeout=self.chat_timeout_seconds,
            )
        if StateGraph is None or START is None or END is None:
            raise ImportError("langgraph is not installed. Install it in the target environment first.")

        graph = StateGraph(AgenticReasoningState)
        graph.add_node("retrieve_knowledge", self.retrieve_knowledge)
        graph.add_node("build_context", self.build_context)
        graph.add_node("retrieve_experience", self.retrieve_experience)
        graph.add_node("initial_reviews", self.initial_reviews)
        graph.add_node("synthesize_report", self.synthesize_report)
        graph.add_node("consultation_round", self.consultation_round)
        graph.add_node("meta_decision", self.meta_decision)
        graph.add_node("finalize_report", self.finalize_report)
        graph.add_node("self_evolve", self.self_evolve)

        graph.add_edge(START, "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "build_context")
        graph.add_edge("build_context", "retrieve_experience")
        graph.add_edge("retrieve_experience", "initial_reviews")
        graph.add_edge("initial_reviews", "synthesize_report")
        graph.add_edge("synthesize_report", "consultation_round")
        graph.add_edge("consultation_round", "meta_decision")
        graph.add_conditional_edges(
            "meta_decision",
            self._route_after_meta_decision,
            {
                "consultation_round": "consultation_round",
                "finalize_report": "finalize_report",
            },
        )
        if self.self_evolution_mode == "inline":
            graph.add_edge("finalize_report", "self_evolve")
            graph.add_edge("self_evolve", END)
        else:
            graph.add_edge("finalize_report", END)
        self.graph = graph.compile()

    def _save_experience_stores(self) -> None:
        if self.split_experience_stores:
            self.meta_experience_store.save()
            for store in self.doctor_experience_stores.values():
                store.save()
            return
        assert self.shared_experience_store is not None
        self.shared_experience_store.save()

    def _experience_store_paths_payload(self) -> dict[str, str]:
        if self.split_experience_stores:
            payload = {"meta_agent": str(self.meta_experience_store.path)}
            for doctor_id, store in self.doctor_experience_stores.items():
                payload[doctor_id] = str(store.path)
            return payload
        return {"shared": str(self.meta_experience_store.path)}

    def _experience_store_size_payload(self) -> dict[str, int]:
        if self.split_experience_stores:
            payload = {"meta_agent": len(self.meta_experience_store.principles)}
            for doctor_id, store in self.doctor_experience_stores.items():
                payload[doctor_id] = len(store.principles)
            return payload
        return {"shared": len(self.meta_experience_store.principles)}

    @staticmethod
    def _prediction_success(
        prediction_context: dict[str, Any] | None,
        consultation_feedback: list[dict[str, str]],
    ) -> bool:
        if prediction_context:
            binary_tasks = prediction_context.get("binary_tasks", [])
            binary_predictions = prediction_context.get("binary_predictions", {})
            binary_targets = prediction_context.get("binary_targets", {})
            comparable_tasks = [task for task in binary_tasks if task in binary_targets]
            if comparable_tasks:
                return all(
                    int(binary_predictions.get(task, 0)) == int(binary_targets.get(task, 0))
                    for task in comparable_tasks
                )
        return not any(item.get("agreement") == "disagree" for item in consultation_feedback)

    @staticmethod
    def _normalize_memory_update_policy(policy: str | int | None) -> str:
        raw = str(policy or "all").strip().lower()
        aliases = {
            "1": "all",
            "always": "all",
            "all_cases": "all",
            "2": "error_only",
            "error": "error_only",
            "errors": "error_only",
            "wrong_only": "error_only",
            "3": "uncertain_or_error",
            "uncertain": "uncertain_or_error",
            "uncertain_and_error": "uncertain_or_error",
        }
        normalized = aliases.get(raw, raw)
        if normalized not in MEMORY_UPDATE_POLICIES:
            return "all"
        return normalized

    @staticmethod
    def _prediction_is_uncertain(
        prediction_context: dict[str, Any] | None,
        low: float = 0.4,
        high: float = 0.6,
    ) -> bool:
        if not prediction_context:
            return False
        probabilities = prediction_context.get("binary_probabilities", {})
        if not isinstance(probabilities, dict):
            return False
        for value in probabilities.values():
            try:
                prob = float(value)
            except (TypeError, ValueError):
                continue
            if low <= prob <= high:
                return True
        return False

    def _should_update_memory(
        self,
        prediction_context: dict[str, Any] | None,
        success: bool,
    ) -> tuple[bool, str]:
        if self.memory_update_policy == "all":
            return True, "policy_all"
        if self.memory_update_policy == "error_only":
            if not success:
                return True, "prediction_error"
            return False, "skipped_correct_prediction"
        if self.memory_update_policy == "uncertain_or_error":
            if not success:
                return True, "prediction_error"
            if self._prediction_is_uncertain(prediction_context):
                return True, "prediction_uncertain"
            return False, "skipped_confident_correct_prediction"
        return True, "policy_all"

    @staticmethod
    def _prediction_context_text(prediction_context: dict[str, Any] | None) -> str:
        if not prediction_context:
            return "Prediction outputs were not provided for memory evolution."
        blocks: list[str] = []
        binary_tasks = prediction_context.get("binary_tasks", [])
        if binary_tasks:
            blocks.append("Binary task results:")
            for task in binary_tasks:
                blocks.append(
                    "- "
                    + " | ".join(
                        [
                            f"task={task}",
                            f"prob={prediction_context.get('binary_probabilities', {}).get(task, 'n/a')}",
                            f"pred={prediction_context.get('binary_predictions', {}).get(task, 'n/a')}",
                            f"target={prediction_context.get('binary_targets', {}).get(task, 'unknown')}",
                        ]
                    )
                )
        regression_tasks = prediction_context.get("regression_tasks", [])
        if regression_tasks:
            blocks.append("Regression task results:")
            for task in regression_tasks:
                blocks.append(
                    "- "
                    + " | ".join(
                        [
                            f"task={task}",
                            f"pred={prediction_context.get('regression_predictions', {}).get(task, 'n/a')}",
                            f"target={prediction_context.get('regression_targets', {}).get(task, 'unknown')}",
                        ]
                    )
                )
        if "success" in prediction_context:
            blocks.append(f"Overall prediction success flag: {prediction_context.get('success')}")
        if prediction_context.get("error_summary"):
            blocks.append(f"Error summary: {prediction_context.get('error_summary')}")
        return "\n".join(blocks) if blocks else "Prediction outputs were provided but empty."

    @staticmethod
    def _build_joint_memory_prompt(
        trajectory_summary: str,
        doctor_summaries: dict[str, str],
        prediction_context: dict[str, Any] | None,
    ) -> str:
        doctor_blocks = [f"[{doctor_id} trajectory]\n{summary}" for doctor_id, summary in doctor_summaries.items()]
        return "\n\n".join(
            [
                f"Completed consultation trajectory:\n{trajectory_summary}",
                f"Prediction evaluation:\n{ColaCareEvolveReasoner._prediction_context_text(prediction_context)}",
                "Doctor-specific trajectories:",
                "\n\n".join(doctor_blocks),
                (
                    'Return strict JSON with exactly four top-level keys: "meta_agent", "doctor_1", '
                    '"doctor_2", and "doctor_3". Each key must map to a list with 1 to 2 memory principles. '
                    'Each principle must be an object with keys "description", "category", and "triples". '
                    '"triples" must be a list of objects with keys "head", "relation", and "tail". '
                    "Base each memory on the final prediction outcome, not only on discussion style. "
                    "If the downstream prediction was wrong, emphasize what this role should change next time. "
                    "Descriptions should be reusable and generalizable, not patient-specific narratives."
                ),
            ]
        )

    def _chat(self, system_prompt: str, user_prompt: str, image_data_url: str = "") -> str:
        if self.backend == "mock":
            return user_prompt[:1000]
        assert self.llm is not None
        last_error: Exception | None = None
        for attempt in range(1, self.chat_max_attempts + 1):
            try:
                if self.chat_throttle_seconds > 0:
                    time.sleep(self.chat_throttle_seconds)
                if image_data_url and HumanMessage is not None:
                    try:
                        response = self.llm.invoke(
                            [
                                HumanMessage(
                                    content=[
                                        {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                                        {"type": "image_url", "image_url": {"url": image_data_url}},
                                    ]
                                )
                            ]
                        )
                        return LangGraphReasoner._extract_content(response.content)
                    except Exception:
                        pass
                response = self.llm.invoke(
                    [
                        ("system", system_prompt),
                        ("human", user_prompt),
                    ]
                )
                return LangGraphReasoner._extract_content(response.content)
            except Exception as exc:
                last_error = exc
                if attempt >= self.chat_max_attempts:
                    break
                wait_seconds = self.chat_retry_wait_seconds * attempt
                time.sleep(wait_seconds)
        assert last_error is not None
        raise last_error

    def retrieve_knowledge(self, state: AgenticReasoningState) -> dict[str, Any]:
        if not self.enable_rag:
            return {"retrieved_docs": [], "rag_enabled": False}
        return {
            "retrieved_docs": self.retriever.search(state["query"], top_k=state.get("top_k", 3)),
            "rag_enabled": True,
        }

    def build_context(self, state: AgenticReasoningState) -> dict[str, Any]:
        return {
            "patient_summary_text": format_patient_summary(state["summary"]),
            "knowledge_text": render_knowledge(state.get("retrieved_docs", []))
            if state.get("rag_enabled")
            else "RAG disabled for this experiment.",
            "current_round": 0,
        }

    def retrieve_experience(self, state: AgenticReasoningState) -> dict[str, Any]:
        hits = self.meta_experience_store.search(state["query"], top_k=self.experience_top_k)
        doctor_hits: dict[str, list[dict[str, Any]]] = {}
        doctor_text: dict[str, str] = {}
        for spec in DOCTOR_SPECS:
            doctor_store = self.doctor_experience_stores[spec["doctor_id"]]
            if self.split_experience_stores:
                scoped_hits = doctor_store.search(state["query"], top_k=self.experience_top_k)
            else:
                scoped_hits = doctor_store.search(
                    state["query"],
                    top_k=self.experience_top_k,
                    owner=spec["doctor_id"],
                    include_global=True,
                )
            doctor_hits[spec["doctor_id"]] = scoped_hits
            doctor_text[spec["doctor_id"]] = _stringify_experience_hits(scoped_hits)
        return {
            "experience_hits": hits,
            "experience_text": _stringify_experience_hits(hits),
            "doctor_experience_hits": doctor_hits,
            "doctor_experience_text": doctor_text,
        }

    def initial_reviews(self, state: AgenticReasoningState) -> dict[str, Any]:
        reviews: list[dict[str, str]] = []
        if self.backend == "mock":
            mock_text = build_mock_reasoning(state["summary"], state.get("retrieved_docs", []))
            for spec in DOCTOR_SPECS:
                reviews.append(
                    {
                        "doctor_id": spec["doctor_id"],
                        "title": spec["title"],
                        "focus": spec["focus"],
                        "review": mock_text,
                    }
                )
            return {"doctor_reviews": reviews}

        admission_record_text = (state.get("note_text") or "unavailable").strip()[:5000]
        image_hint = state.get("image_path") or "unavailable"
        doctor_memory = state.get("doctor_experience_text", {})
        for spec in DOCTOR_SPECS:
            print(
                f"[multiagent] initial_review {spec['doctor_id']} {state['summary'].get('sample_id', '')}",
                flush=True,
            )
            scope = DOCTOR_PROMPT_SCOPES.get(spec["doctor_id"], {})
            focused_patient_prompt = build_focused_clinical_reasoning_user_prompt(
                summary=state["summary"],
                admission_text_raw=admission_record_text,
                static_keys=scope.get("static_keys"),
                dynamic_keys=scope.get("dynamic_keys"),
            )
            prompt_parts = [
                f"Role: {spec['title']}",
                spec["focus"],
                f"External knowledge status:\n{state['knowledge_text']}",
                f"Your memory principles:\n{doctor_memory.get(spec['doctor_id'], 'No prior experience principles retrieved.')}",
                focused_patient_prompt,
            ]
            if image_hint and image_hint != "unavailable":
                prompt_parts.append(f"Image context: {image_hint}")
            prompt_parts.append(
                'Return strict JSON with keys "structured_summary", "text_cleaned", '
                '"reasoning_dimensions", and "overall_assessment". '
                "Summarize current objective status only. Do not output risk level, mortality probability, "
                "survival likelihood, or 12/24/48-hour outcome."
            )
            prompt = "\n\n".join(prompt_parts)
            review = self._chat(
                system_prompt=COLACARE_SYSTEM_PROMPT,
                user_prompt=prompt,
                image_data_url=str(state.get("image_data_url", "")),
            )
            reviews.append(
                {
                    "doctor_id": spec["doctor_id"],
                    "title": spec["title"],
                    "focus": spec["focus"],
                    "system_prompt": COLACARE_SYSTEM_PROMPT,
                    "user_prompt": prompt,
                    "review": review,
                }
            )
        return {"doctor_reviews": reviews}

    def synthesize_report(self, state: AgenticReasoningState) -> dict[str, Any]:
        if self.backend == "mock":
            final_report = build_mock_reasoning(state["summary"], state.get("retrieved_docs", []))
            return {
                "meta_system_prompt": META_SYSTEM_PROMPT,
                "meta_user_prompt": "mock synthesis",
                "meta_report": final_report,
            }
        print(f"[multiagent] meta_synthesize {state['summary'].get('sample_id', '')}", flush=True)
        review_block = "\n\n".join(
            [
                f"{item['title']} ({item['doctor_id']}):\n{item['review']}"
                for item in state.get("doctor_reviews", [])
            ]
        )
        prompt = "\n\n".join(
            [
                f"Patient summary:\n{state['patient_summary_text']}",
                f"Doctor initial reviews:\n{review_block}",
                f"External knowledge status:\n{state['knowledge_text']}",
                (
                    'Return strict JSON with keys "structured_summary", "text_cleaned", '
                    '"reasoning_dimensions", "overall_assessment", and "agreements_disagreements". '
                    "Integrate agreements and disagreements, but summarize only current disease state. "
                    "Do not output any predictive conclusion."
                ),
            ]
        )
        report = self._chat(META_SYSTEM_PROMPT, prompt, image_data_url=str(state.get("image_data_url", "")))
        return {
            "meta_system_prompt": META_SYSTEM_PROMPT,
            "meta_user_prompt": prompt,
            "meta_report": report,
        }

    def consultation_round(self, state: AgenticReasoningState) -> dict[str, Any]:
        if self.backend == "mock":
            feedback = []
            for item in state.get("doctor_reviews", []):
                feedback.append(
                    {
                        "doctor_id": item["doctor_id"],
                        "agreement": "agree",
                        "comment": "The synthesized report is acceptable.",
                    }
                )
            return {"consultation_feedback": feedback}

        feedback: list[dict[str, str]] = []
        for item in state.get("doctor_reviews", []):
            print(
                f"[multiagent] consultation_feedback {item['doctor_id']} {state['summary'].get('sample_id', '')}",
                flush=True,
            )
            prompt = "\n\n".join(
                [
                    f"Role: {item['title']}",
                    f"Your initial review:\n{item['review']}",
                    f"Current MetaAgent report:\n{state.get('meta_report', '')}",
                    f"External knowledge status:\n{state['knowledge_text']}",
                    (
                        'Return strict JSON with keys "agreement", "comment", and "evidence". '
                        'Set "agreement" to "agree" or "disagree". Keep the comment concise. '
                        "Do not add mortality risk predictions."
                    ),
                ]
            )
            raw = self._chat(COLACARE_SYSTEM_PROMPT, prompt)
            try:
                parsed = _json_object_or_fallback(
                    _safe_json_loads(raw),
                    {"agreement": "agree", "comment": "", "evidence": ""},
                )
            except Exception:
                lowered = raw.lower()
                parsed = {
                    "agreement": "disagree" if "disagree" in lowered else "agree",
                    "comment": raw.strip(),
                    "evidence": "",
                }
            feedback.append(
                {
                    "doctor_id": item["doctor_id"],
                    "agreement": str(parsed.get("agreement", "agree")).strip().lower(),
                    "comment": str(parsed.get("comment", "")).strip(),
                    "evidence": str(parsed.get("evidence", "")).strip(),
                }
            )
        return {"consultation_feedback": feedback}

    def meta_decision(self, state: AgenticReasoningState) -> dict[str, Any]:
        current_round = int(state.get("current_round", 0)) + 1
        feedback = state.get("consultation_feedback", [])
        disagreements = [item for item in feedback if item.get("agreement") == "disagree"]
        if self.backend == "mock":
            return {"current_round": current_round, "continue_discussion": False}
        if not disagreements or current_round >= self.max_consult_rounds:
            return {"current_round": current_round, "continue_discussion": False}
        feedback_text = "\n\n".join(
            [
                "\n".join(
                    [
                        f"Doctor: {item['doctor_id']}",
                        f"Agreement: {item['agreement']}",
                        f"Comment: {item['comment']}",
                        f"Evidence: {item.get('evidence', '')}",
                    ]
                )
                for item in feedback
            ]
        )
        prompt = "\n\n".join(
            [
                f"Current report:\n{state.get('meta_report', '')}",
                f"Doctor feedback:\n{feedback_text}",
                (
                    'Return strict JSON with keys "action", "updated_report", and "reason". '
                    'Choose action "continue" or "stop". If continuing, revise the report. '
                    "Do not add mortality risk predictions."
                ),
            ]
        )
        raw = self._chat(META_SYSTEM_PROMPT, prompt)
        try:
            parsed = _json_object_or_fallback(
                _safe_json_loads(raw),
                {"action": "stop", "updated_report": raw.strip(), "reason": "non-object JSON fallback"},
            )
        except Exception:
            parsed = {
                "action": "continue",
                "updated_report": raw.strip(),
                "reason": "fallback parse",
            }
        action = str(parsed.get("action", "stop")).strip().lower()
        updated_report = str(parsed.get("updated_report", state.get("meta_report", ""))).strip()
        return {
            "current_round": current_round,
            "continue_discussion": action == "continue" and current_round < self.max_consult_rounds,
            "meta_report": updated_report or state.get("meta_report", ""),
        }

    @staticmethod
    def _route_after_meta_decision(state: AgenticReasoningState) -> str:
        if state.get("continue_discussion"):
            return "consultation_round"
        return "finalize_report"

    def finalize_report(self, state: AgenticReasoningState) -> dict[str, Any]:
        report = str(state.get("meta_report", "")).strip()
        return {
            "final_report": report,
            "reasoning_text": report,
        }

    def self_evolve(self, state: AgenticReasoningState) -> dict[str, Any]:
        if self.self_evolution_mode == "post_prediction":
            paths_payload = self._experience_store_paths_payload()
            return {
                "evolution_result": {
                    "updated_principle_ids": [],
                    "experience_store_path": paths_payload.get("shared", paths_payload.get("meta_agent", "")),
                    "experience_store_paths": paths_payload,
                    "deferred": True,
                    "mode": "post_prediction",
                }
            }
        return self.evolve_after_prediction(state, prediction_context=None)

    def evolve_after_prediction(
        self,
        state: AgenticReasoningState,
        prediction_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieved_ids = [str(item.get("id", "")) for item in state.get("experience_hits", []) if item.get("id")]
        retrieved_ids = list(dict.fromkeys(item for item in retrieved_ids if item))
        doctor_retrieved_ids: dict[str, list[str]] = {}
        for spec in DOCTOR_SPECS:
            doctor_id = spec["doctor_id"]
            doctor_ids = [
                str(item.get("id", ""))
                for item in state.get("doctor_experience_hits", {}).get(doctor_id, [])
                if item.get("id")
            ]
            doctor_retrieved_ids[doctor_id] = list(dict.fromkeys(item for item in doctor_ids if item))

        success = self._prediction_success(prediction_context, state.get("consultation_feedback", []))
        self.meta_experience_store.mark_usage(retrieved_ids, success=success)
        for doctor_id, doctor_ids in doctor_retrieved_ids.items():
            self.doctor_experience_stores[doctor_id].mark_usage(doctor_ids, success=success)
        should_update_memory, memory_update_reason = self._should_update_memory(prediction_context, success)
        if not should_update_memory:
            self._save_experience_stores()
            paths_payload = self._experience_store_paths_payload()
            size_payload = self._experience_store_size_payload()
            return {
                "evolution_result": {
                    "updated_principle_ids": [],
                    "doctor_updated_principle_ids": {},
                    "experience_store_path": paths_payload.get("shared", paths_payload.get("meta_agent", "")),
                    "experience_store_paths": paths_payload,
                    "experience_size": sum(size_payload.values()),
                    "experience_sizes": size_payload,
                    "success": success,
                    "mode": self.self_evolution_mode,
                    "memory_update_policy": self.memory_update_policy,
                    "memory_update_skipped": True,
                    "memory_update_reason": memory_update_reason,
                }
            }
        if not self.enable_self_evolution:
            self._save_experience_stores()
            paths_payload = self._experience_store_paths_payload()
            return {
                "evolution_result": {
                    "updated_principle_ids": [],
                    "experience_store_path": paths_payload.get("shared", paths_payload.get("meta_agent", "")),
                    "experience_store_paths": paths_payload,
                    "mode": self.self_evolution_mode,
                    "memory_update_policy": self.memory_update_policy,
                    "memory_update_skipped": True,
                    "memory_update_reason": "self_evolution_disabled",
                }
            }

        review_map = {item.get("doctor_id", ""): item for item in state.get("doctor_reviews", [])}
        feedback_map = {item.get("doctor_id", ""): item for item in state.get("consultation_feedback", [])}
        doctor_summaries: dict[str, str] = {}
        for spec in DOCTOR_SPECS:
            doctor_id = spec["doctor_id"]
            review_item = review_map.get(doctor_id, {})
            feedback_item = feedback_map.get(doctor_id, {})
            doctor_summaries[doctor_id] = "\n\n".join(
                [
                    f"Doctor role: {spec['title']}",
                    f"Doctor focus: {spec['focus']}",
                    f"Patient summary:\n{state.get('patient_summary_text', '')}",
                    f"Initial review:\n{review_item.get('review', '')}",
                    f"Feedback:\nAgreement={feedback_item.get('agreement', '')}\nComment={feedback_item.get('comment', '')}\nEvidence={feedback_item.get('evidence', '')}",
                    f"Final meta report:\n{state.get('final_report', '')}",
                ]
            ).strip()

        trajectory_summary = "\n\n".join(
            [
                f"Patient summary:\n{state.get('patient_summary_text', '')}",
                f"Final report:\n{state.get('final_report', '')}",
                "Doctor feedback:",
                "\n".join(
                    [
                        f"- {item.get('doctor_id')}: {item.get('agreement')} | {item.get('comment')}"
                        for item in state.get("consultation_feedback", [])
                    ]
                ),
            ]
        ).strip()

        if self.backend == "mock":
            distilled_payload: dict[str, list[dict[str, Any]]] = {
                "meta_agent": [
                    {
                        "description": "Meta memory should be grounded in whether the downstream prognosis head matched the true label.",
                        "category": "meta_outcome_memory",
                        "triples": [
                            {"head": "final prediction", "relation": "must_be_checked_against", "tail": "true label"},
                        ],
                    }
                ],
                "doctor_1": [
                    {
                        "description": "The hemodynamic and respiratory role should revisit perfusion and oxygenation cues when the final prediction is wrong.",
                        "category": "role_memory",
                        "triples": [
                            {
                                "head": "doctor_1",
                                "relation": "focuses_on",
                                "tail": "perfusion and oxygenation error patterns",
                            },
                        ],
                    }
                ],
                "doctor_2": [
                    {
                        "description": "The organ dysfunction role should refine how laboratory trajectories are weighed against the final label.",
                        "category": "role_memory",
                        "triples": [
                            {"head": "doctor_2", "relation": "focuses_on", "tail": "laboratory trajectory mismatch"},
                        ],
                    }
                ],
                "doctor_3": [
                    {
                        "description": "The history and treatment role should reassess how baseline reserve and treatment burden affect downstream risk estimation.",
                        "category": "role_memory",
                        "triples": [
                            {"head": "doctor_3", "relation": "focuses_on", "tail": "reserve and treatment weighting"},
                        ],
                    }
                ],
            }
        else:
            prompt = self._build_joint_memory_prompt(trajectory_summary, doctor_summaries, prediction_context)
            raw = self._chat(EVOLUTION_SYSTEM_PROMPT, prompt)
            try:
                distilled_payload = _safe_json_loads(raw)
            except Exception:
                distilled_payload = {}
            if not isinstance(distilled_payload, dict):
                distilled_payload = {}

        meta_principles = distilled_payload.get("meta_agent", [])
        if not isinstance(meta_principles, list):
            meta_principles = []
        updated_ids = self.meta_experience_store.integrate(
            principles=[item for item in meta_principles if isinstance(item, dict)],
            case_id=str(state["summary"].get("sample_id", "synthetic_case")),
            trajectory_summary=trajectory_summary,
            success=success,
            owner="meta_agent" if self.split_experience_stores else "global",
        )
        doctor_updated: dict[str, list[str]] = {}
        for spec in DOCTOR_SPECS:
            doctor_id = spec["doctor_id"]
            doctor_principles = distilled_payload.get(doctor_id, [])
            if not isinstance(doctor_principles, list):
                doctor_principles = []
            doctor_updated[doctor_id] = self.doctor_experience_stores[doctor_id].integrate(
                principles=[item for item in doctor_principles if isinstance(item, dict)],
                case_id=str(state["summary"].get("sample_id", "synthetic_case")),
                trajectory_summary=doctor_summaries[doctor_id],
                success=success,
                owner=doctor_id,
            )
        self._save_experience_stores()
        paths_payload = self._experience_store_paths_payload()
        size_payload = self._experience_store_size_payload()
        return {
            "evolution_result": {
                "updated_principle_ids": updated_ids,
                "doctor_updated_principle_ids": doctor_updated,
                "experience_store_path": paths_payload.get("shared", paths_payload.get("meta_agent", "")),
                "experience_store_paths": paths_payload,
                "experience_size": sum(size_payload.values()),
                "experience_sizes": size_payload,
                "success": success,
                "mode": self.self_evolution_mode,
                "memory_update_policy": self.memory_update_policy,
                "memory_update_skipped": False,
                "memory_update_reason": memory_update_reason,
                "memory_prompt": self._build_joint_memory_prompt(
                    trajectory_summary,
                    doctor_summaries,
                    prediction_context,
                ),
            }
        }

    def invoke(
        self,
        case: dict[str, object],
        top_k: int = 3,
        note_text: str = "",
        image_data_url: str = "",
        image_path: str = "",
    ) -> dict[str, Any]:
        state: AgenticReasoningState = {
            "summary": dict(case["summary"]),
            "query": str(case["query"]),
            "top_k": top_k,
            "note_text": note_text,
            "image_data_url": image_data_url,
            "image_path": image_path,
        }
        return self.graph.invoke(state)


def build_reasoner(
    workflow: str,
    docs_dir: str | Path | list[str] | list[Path],
    api_key: str | None,
    model_name: str,
    base_url: str,
    backend: str = "deepseek",
    temperature: float = 0.0,
    experience_store_path: str | Path = "data/reasoning/colacare_evolver_experience.json",
    experience_store_paths: dict[str, str | Path] | None = None,
    experience_top_k: int = 3,
    max_consult_rounds: int = 2,
    enable_self_evolution: bool = True,
    self_evolution_mode: str = "inline",
    memory_update_policy: str = "all",
    enable_rag: bool = False,
):
    normalized = workflow.strip().lower()
    if normalized in {"single_agent", "langgraph_single", "baseline"}:
        return LangGraphReasoner(
            docs_dir=docs_dir,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            backend=backend,
            temperature=temperature,
            enable_rag=enable_rag,
        )
    if normalized in {"colacare_evolver", "colacare", "multi_agent"}:
        return ColaCareEvolveReasoner(
            docs_dir=docs_dir,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            backend=backend,
            temperature=temperature,
            experience_store_path=experience_store_path,
            experience_store_paths=experience_store_paths,
            experience_top_k=experience_top_k,
            max_consult_rounds=max_consult_rounds,
            enable_self_evolution=enable_self_evolution,
            self_evolution_mode=self_evolution_mode,
            memory_update_policy=memory_update_policy,
            enable_rag=enable_rag,
        )
    raise ValueError(f"Unsupported reasoning workflow: {workflow}")
