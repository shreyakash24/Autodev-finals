"""
Coding Agents (Frontend, Backend, Database)
Template Fallbacks.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import re

from crewai import Agent, Task as CrewTask
from ..models import GeneratedArtifact, AgentType
from ..utils.llm_config import get_llm
from ..utils.code_cleaner import CodeCleaner

class BaseCodingAgent:
    """Base class upgraded with centralized execution and context management."""
    
    def __init__(self, agent_type: AgentType, role: str, goal: str, backstory: str):
        self.agent_type = agent_type
        self.llm = get_llm(temperature=0.2)
        self.crew_agent = None
        if self.llm:
            try:
                self.crew_agent = Agent(
                    role=role,
                    goal=goal,
                    backstory=backstory,
                    verbose=True,
                    allow_delegation=False,
                    llm=self.llm
                )
            except Exception as e:
                print(f"Warning: Failed to create CrewAI agent: {e}")

    def _execute_llm_task(self, description: str, expected: str) -> str:
        """Executes LLM task and scrubs the output of markdown."""
        if not self.crew_agent:
            return ""
        try:
            task = CrewTask(description=description, agent=self.crew_agent, expected_output=expected)
            result = task.execute_sync()
            return CodeCleaner.clean(str(result))
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            return ""

    def _inject_context(self, requirements: Dict[str, Any]) -> str:
        """Injects legacy patterns and refined prompts into the LLM context."""
        context = "\n--- CONTEXTUAL CONSTRAINTS ---\n"
        if "_legacy_patterns" in requirements:
            context += f"LEGACY CODEBASE PATTERNS: {requirements['_legacy_patterns']}\n"
        if "_refined_context" in requirements:
            context += f"REFINED SPECIFICATION: {requirements['_refined_context']}\n"
        return context

    def _create_artifact(self, file_path: str, content: str, artifact_type: str, language: str, documentation: str = "") -> GeneratedArtifact:
        return GeneratedArtifact(
            file_path=file_path,
            content=content,
            artifact_type=artifact_type,
            language=language,
            agent_type=self.agent_type,
            documentation=documentation
        )

    def _to_pascal_case(self, text: str) -> str:
        words = re.sub(r'[^a-zA-Z0-9\s_]', '', text).replace('_', ' ').split()
        return ''.join(word.capitalize() for word in words)


class FrontendCodingAgent(BaseCodingAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.FRONTEND_CODER,
            role="Senior Frontend Developer",
            goal="Generate high-quality React components with accessibility and legacy alignment",
            backstory="You are an expert at React 18, TypeScript, and aligning new UI with legacy architectures."
        )

    def generate_component_scaffold(self, requirements: Dict[str, Any], tech_stack: str = "React") -> List[GeneratedArtifact]:
        artifacts = []
        user_stories = requirements.get('user_stories', [])
        context = self._inject_context(requirements)

        # Attempt LLM Generation
        if self.crew_agent and user_stories:
            app_prompt = f"Generate a React App.tsx with Routing and Error Boundaries based on these stories: {user_stories}. {context}"
            app_code = self._execute_llm_task(app_prompt, "Pure React code for App.tsx")
            if app_code:
                artifacts.append(self._create_artifact("src/App.tsx", app_code, "component", "typescript", "App Entry Point"))
            
            for story in user_stories[:3]:
                name = self._to_pascal_case(story.get('title', 'Feature'))
                comp_prompt = f"Generate a React TypeScript component for: {story.get('title')}. {context}"
                code = self._execute_llm_task(comp_prompt, f"React code for {name}.tsx")
                if code:
                    artifacts.append(self._create_artifact(f"src/components/{name}.tsx", code, "component", "typescript"))

        # Fallback to Templates if artifacts are empty
        if not artifacts:
            artifacts.append(self._create_artifact("src/App.tsx", self._generate_app_component(requirements), "component", "typescript"))
            artifacts.append(self._create_artifact("src/router/index.tsx", self._generate_router(requirements), "router", "typescript"))
        
        return artifacts

    def generate_forms(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        prompt = f"Generate a reusable React useForm hook and Form component with validation. {self._inject_context(requirements)}"
        code = self._execute_llm_task(prompt, "TypeScript Form hooks")
        
        if code:
            return [self._create_artifact("src/hooks/useForm.ts", code, "hook", "typescript")]
        
        # Fallback to original templates
        return [
            self._create_artifact("src/components/Form/Form.tsx", self._generate_generic_form(), "component", "typescript"),
            self._create_artifact("src/hooks/useForm.ts", self._generate_custom_hooks(), "hook", "typescript")
        ]

    def add_accessibility(self, components: List[GeneratedArtifact]) -> List[GeneratedArtifact]:
        a11y_utils = self._generate_a11y_utils()
        return [self._create_artifact("src/utils/accessibility.tsx", a11y_utils, "utility", "typescript")]

    # --- Fallback Templates ---
    def _generate_app_component(self, r): return """/* App Template */\nimport React from 'react';\nexport default function App() { return <div>App</div>; }"""
    def _generate_router(self, r): return """/* Router Template */\nimport { BrowserRouter } from 'react-router-dom';"""
    def _generate_generic_form(self): return """/* Form Template */"""
    def _generate_custom_hooks(self): return """/* useForm Template */"""
    def _generate_a11y_utils(self): return """/* A11y Utils */"""


class BackendCodingAgent(BaseCodingAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.BACKEND_CODER,
            role="Senior Backend Developer",
            goal="Generate clean, secure FastAPI REST APIs",
            backstory="Expert in Python, Pydantic, and secure API design."
        )

    def generate_api_contracts(self, requirements: Dict[str, Any], tech_stack: str = "FastAPI") -> List[GeneratedArtifact]:
        artifacts = []
        context = self._inject_context(requirements)

        if self.crew_agent:
            prompt = f"Generate FastAPI routes and Pydantic schemas for: {requirements.get('user_stories')}. {context}"
            code = self._execute_llm_task(prompt, "Python FastAPI code")
            if code:
                artifacts.append(self._create_artifact("app/api/routes.py", code, "routes", "python"))

        if not artifacts:
            artifacts.append(self._create_artifact("app/main.py", self._generate_main_app(), "application", "python"))
            artifacts.append(self._create_artifact("app/schemas.py", self._generate_schemas(requirements), "schemas", "python"))
        
        return artifacts

    def generate_services(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        prompt = f"Generate backend service business logic for: {requirements}. {self._inject_context(requirements)}"
        code = self._execute_llm_task(prompt, "Python Service Layer")
        if code:
            return [self._create_artifact("app/services/business_logic.py", code, "service", "python")]
        return [self._create_artifact("app/services/base.py", self._generate_base_service(), "service", "python")]

    def generate_controllers(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        return [self._create_artifact("app/controllers/base.py", self._generate_controller(), "controller", "python")]

    # --- Fallback Templates ---
    def _generate_main_app(self): return '"""Main App"""'
    def _generate_schemas(self, r): return '"""Schemas"""'
    def _generate_base_service(self): return '"""Service"""'
    def _generate_controller(self): return '"""Controller"""'


class DatabaseCodingAgent(BaseCodingAgent):
    def __init__(self):
        super().__init__(
            agent_type=AgentType.DATABASE_CODER,
            role="Database Developer",
            goal="Generate optimized SQLAlchemy schemas and migrations",
            backstory="Expert in database normalization and SQLAlchemy 2.0."
        )

    def generate_schema(self, requirements: Dict[str, Any], tech_stack: str = "PostgreSQL") -> List[GeneratedArtifact]:
        artifacts = []
        context = self._inject_context(requirements)

        if self.crew_agent:
            prompt = f"Generate SQLAlchemy 2.0 models for: {requirements.get('user_stories')}. {context}"
            code = self._execute_llm_task(prompt, "SQLAlchemy Python code")
            if code:
                artifacts.append(self._create_artifact("app/models/base.py", code, "model", "python"))

        if not artifacts:
            artifacts.append(self._create_artifact("app/models/base.py", self._generate_models(requirements), "model", "python"))
        
        artifacts.append(self._create_artifact("app/database.py", self._generate_db_config(), "config", "python"))
        return artifacts

    def generate_orm_models(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        return [self._create_artifact("app/models/item.py", self._generate_item_model(), "model", "python")]

    # --- Fallback Templates ---
    def _generate_models(self, r): return '"""Models"""'
    def _generate_db_config(self): return '"""DB Config"""'
    def _generate_item_model(self): return '"""Item Model"""'