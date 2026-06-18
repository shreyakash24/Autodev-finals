"""
CodeGenerationCrew: The central orchestrator.
"""

from typing import List, Dict, Any, Optional
import os
import json

from .agents import (
    ADOConnectorAgent, OrchestratorAgent,
    FrontendCodingAgent, BackendCodingAgent, DatabaseCodingAgent,
    TestingAgent, LegacyAnalyzerAgent, PromptRefinerAgent, MonitoringAgent
)
from .models import CanonicalSpec, PipelineState, Task, TaskStatus, GeneratedArtifact, AgentType, UserStory
from .utils.code_cleaner import CodeCleaner

class CodeGenerationCrew:
    
    def __init__(self, auto_mode: bool = True, ado_config: Optional[Dict[str, str]] = None):
        # Initialize Agents
        self.ado_agent = ADOConnectorAgent()
        self.orchestrator = OrchestratorAgent()
        self.frontend_agent = FrontendCodingAgent()
        self.backend_agent = BackendCodingAgent()
        self.database_agent = DatabaseCodingAgent()
        self.testing_agent = TestingAgent()
        self.legacy_agent = LegacyAnalyzerAgent()
        self.prompt_agent = PromptRefinerAgent()
        self.monitoring_agent = MonitoringAgent()
        
        # Internal State
        self.current_spec: Optional[CanonicalSpec] = None
        self.current_pipeline: Optional[PipelineState] = None
        self.legacy_analysis: Optional[Dict[str, Any]] = None
        
        # Config
        self.auto_mode = auto_mode
        self.ado_config = ado_config or {}
        
        self._setup_monitoring()

    def _setup_monitoring(self):
        """Standard console logger for events."""
        def log_event(event):
            print(f"[{event.timestamp.strftime('%H:%M:%S')}] {event.event_type.value.upper()}: {event.data}")
        self.monitoring_agent.subscribe(log_event)

    def process_requirements(self, data: str, data_format: str = 'json', repo_path: Optional[str] = None) -> CanonicalSpec:
        """Processes and refines raw ADO data into a canonical specification."""
        
        # 1. Analyze Legacy Repo first to set the 'Standard'
        if self.auto_mode and repo_path and os.path.exists(repo_path):
            print("[Intelligence] Analyzing legacy repository for architectural context...")
            self.legacy_analysis = self.analyze_legacy(repo_path)
        
        # 2. Parse raw data
        self.current_spec = self.ado_agent.process(data, data_format)
        
        # 3. Refine Prompts (The 'Sanity Check')
        if self.auto_mode:
            print("[Intelligence] Scrubbing user stories for ambiguity...")
            for story in self.current_spec.user_stories:
                if story.description:
                    refinement = self.prompt_agent.refine_prompt(story.description, criteria=story.acceptance_criteria)
                    if refinement.confidence_score > 0.7:
                        story.description = refinement.refined_prompt
                        print(f"  ✓ Refined Story {story.id} (Confidence: {refinement.confidence_score:.0%})")
        
        return self.current_spec

    def build_pipeline(self) -> PipelineState:
        """Prepares the task sequence based on the spec."""
        if not self.current_spec:
            raise ValueError("No specification available. Process requirements first.")
        
        self.current_pipeline = self.orchestrator.build_pipeline(self.current_spec)
        self.monitoring_agent.set_pipeline(self.current_pipeline)
        return self.current_pipeline

    def execute_pipeline(self, parallel: bool = True) -> PipelineState:
        """Runs the pipeline with Intelligence Context Injection."""
        if not self.current_pipeline:
            raise ValueError("Pipeline not built.")

        groups = self.orchestrator.get_parallel_groups() if parallel else [[t] for t in self.current_pipeline.tasks]
        
        for group in groups:
            for task in group:
                self._execute_task(task)
        
        # Add project plumbing (README, package.json, etc.)
        self._add_project_structure_files()
        
        # Handle Auto-Commit if configured
        if self.auto_mode and self.ado_config.get('org_url'):
            self._auto_commit_to_ado()
            
        return self.current_pipeline

    def _execute_task(self, task: Task) -> None:
        """The heart of the generation: Merges context, generates, and scrubs code."""
        self.orchestrator.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        
        try:
            # --- 1. PREPARE CONTEXT ---
            requirements = task.input_data.get('requirements', {})
            requirements['user_stories'] = task.input_data.get('user_stories', [])
            
            if self.auto_mode:
                # Inject Legacy Knowledge
                if self.legacy_analysis:
                    requirements['_legacy_patterns'] = self.legacy_analysis.get('conventions', [])
                
                # Dynamic Prompt Refinement per Task
                refinement = self.prompt_agent.refine_prompt(task.description)
                requirements['_refined_context'] = refinement.refined_prompt
            
            # --- 2. GENERATE ---
            artifacts = []
            agent_map = {
                AgentType.FRONTEND_CODER: self.frontend_agent,
                AgentType.BACKEND_CODER: self.backend_agent,
                AgentType.DATABASE_CODER: self.database_agent,
                AgentType.TESTING: self.testing_agent
            }

            agent = agent_map.get(task.agent_type)
            if not agent:
                raise ValueError(f"No agent configured for {task.agent_type}")

            # Route based on agent capabilities
            if task.agent_type == AgentType.FRONTEND_CODER:
                artifacts = agent.generate_component_scaffold(requirements)
                artifacts.extend(agent.generate_forms(requirements))
            elif task.agent_type == AgentType.BACKEND_CODER:
                artifacts = agent.generate_api_contracts(requirements)
                artifacts.extend(agent.generate_services(requirements))
            elif task.agent_type == AgentType.DATABASE_CODER:
                artifacts = agent.generate_schema(requirements)
            elif task.agent_type == AgentType.TESTING:
                if task.input_data.get('task_type') == 'test_plan':
                    # Special case: Test Plan Object
                    test_plan = agent.generate_test_plan(
                        self.current_spec.user_stories, 
                        self.current_pipeline.artifacts, 
                        self.current_spec.project_name or "GeneratedProject"
                    )
                    self.current_pipeline.test_plans.append(test_plan)
                else:
                    artifacts = agent.generate_unit_tests(requirements, self.current_pipeline.artifacts)

            # --- 3. SCRUB & ADD ---
            for artifact in artifacts:
                # CRITICAL: Strip markdown garbage
                artifact.content = CodeCleaner.clean(artifact.content)
                self.orchestrator.add_artifact(artifact)

            self.orchestrator.update_task_status(task.id, TaskStatus.COMPLETED)
            
        except Exception as e:
            print(f"[ERROR] Task {task.name} failed: {str(e)}")
            self.orchestrator.update_task_status(task.id, TaskStatus.FAILED, error_message=str(e))

    def _add_project_structure_files(self):
        """Generates standard boilerplate like README, requirements.txt, and .gitignore."""
        print("[Pipeline] Finalizing project structure...")
        # Add a README
        readme = f"# {self.current_spec.project_name or 'Generated Project'}\nGenerated via Azure CodeAgents.\n"
        self.orchestrator.add_artifact(GeneratedArtifact(
            file_path="README.md", content=readme, artifact_type="docs", language="markdown"
        ))
        # Add Requirements
        self.orchestrator.add_artifact(GeneratedArtifact(
            file_path="requirements.txt", content="fastapi\nuvicorn\nsqlalchemy\npytest", 
            artifact_type="config", language="text"
        ))

    def _auto_commit_to_ado(self):
        """Publishes artifacts to Azure DevOps automatically."""
        print("[AutoMode] Attempting auto-commit to Azure DevOps...")
        try:
            temp_ado = ADOConnectorAgent(ado_url=self.ado_config['org_url'], pat=self.ado_config['pat'])
            temp_ado.project = self.ado_config['project']
            
            result = temp_ado.publish_to_azure_devops(
                pipeline_state=self.current_pipeline,
                repo_name=self.ado_config.get('repo_name', 'generated-code'),
                branch=self.ado_config.get('branch', 'refs/heads/generated-code')
            )
            self.current_pipeline.commit_result = result
            print(f"[AutoMode] ✓ ADO Commit Status: {result.get('success')}")
        except Exception as e:
            print(f"[AutoMode] ADO Commit Failed: {e}")

    def analyze_legacy(self, repo_path: str) -> Dict[str, Any]:
        """Analysis helper."""
        analysis = self.legacy_agent.analyze_repository(repo_path)
        self.legacy_analysis = analysis.to_dict()
        return self.legacy_analysis

    def run_tests(self, test_type: str = 'unit') -> Dict[str, Any]:
        """Runs generated tests via the testing agent."""
        from .models import TestType
        type_enum = TestType.UNIT if test_type == 'unit' else TestType.INTEGRATION
        report = self.testing_agent.run_tests('tests/', type_enum)
        self.orchestrator.add_test_report(report)
        return report.to_dict()