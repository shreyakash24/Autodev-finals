"""
Coordinating Agent (Orchestrator)
Builds the pipeline from the canonical spec: subtasks for UI, API, schema, tests, and integration steps.
Creates a dependency graph and enables parallel branches where safe.
Enforces quality gates: blocks merges if tests fail or coverage thresholds are unmet.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import os
from collections import deque

from crewai import Agent, Crew, Process

from ..utils.llm_config import get_llm

from ..models import (
    CanonicalSpec, Task, TaskStatus, AgentType, 
    PipelineState, TestReport, GeneratedArtifact
)


class OrchestratorAgent:
    """
    Central orchestrator that coordinates all other agents.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Orchestrator Agent                               │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Pipeline Builder                               │ │
    │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │ │
    │  │  │ Spec Parser │─▶│ Task Creator │─▶│ Dependency Graph Builder  │  │ │
    │  │  └─────────────┘  └──────────────┘  └───────────────────────────┘  │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Task Scheduler                                 │ │
    │  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐ │ │
    │  │  │Parallel Runner│  │ Quality Gates │  │ Agent Task Assignment   │ │ │
    │  │  └───────────────┘  └───────────────┘  └─────────────────────────┘ │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      State Manager                                  │ │
    │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────────┐   │ │
    │  │  │ Artifacts  │  │ Test Rpts  │  │ Event Logger               │   │ │
    │  │  └────────────┘  └────────────┘  └────────────────────────────┘   │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, coverage_threshold: float = 80.0, max_parallel: int = 4):
        """
        Initialize the Orchestrator Agent.
        
        Args:
            coverage_threshold: Minimum test coverage required (percentage)
            max_parallel: Maximum number of parallel tasks
        """
        self.coverage_threshold = coverage_threshold
        self.max_parallel = max_parallel
        self.pipeline_state: Optional[PipelineState] = None
        self.event_callbacks: List[callable] = []
        
        # Initialize CrewAI components
        self.llm = get_llm(temperature=0.1)
        
        self.crew_agent = Agent(
            role="Project Orchestrator",
            goal="Coordinate code generation and testing pipeline",
            backstory="""You are an expert project coordinator who understands
            software development workflows. You create optimal task sequences
            and ensure quality gates are met before proceeding.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        ) if self.llm else None
    
    def build_pipeline(self, spec: CanonicalSpec) -> PipelineState:
        """
        Build a complete pipeline from the canonical specification.
        
        Args:
            spec: Canonical specification from ADO Connector
            
        Returns:
            PipelineState with all tasks and dependencies
        """
        self.pipeline_state = PipelineState(spec=spec)
        tasks = []
        
        # Phase 1: Database Schema (no dependencies)
        db_tasks = self._create_database_tasks(spec)
        tasks.extend(db_tasks)
        
        # Phase 2: Backend (depends on database)
        backend_tasks = self._create_backend_tasks(spec, db_tasks)
        tasks.extend(backend_tasks)
        
        # Phase 3: Frontend (can run parallel with backend in some cases)
        frontend_tasks = self._create_frontend_tasks(spec, backend_tasks)
        tasks.extend(frontend_tasks)
        
        # Phase 4: Unit Tests (depends on code generation)
        unit_test_tasks = self._create_unit_test_tasks(
            spec, db_tasks + backend_tasks + frontend_tasks
        )
        tasks.extend(unit_test_tasks)
        
        # Phase 5: Integration Tests (depends on unit tests)
        integration_test_tasks = self._create_integration_test_tasks(
            spec, unit_test_tasks
        )
        tasks.extend(integration_test_tasks)
        
        # Phase 6: E2E Tests (depends on integration tests)
        e2e_test_tasks = self._create_e2e_test_tasks(
            spec, integration_test_tasks
        )
        tasks.extend(e2e_test_tasks)
        
        # Phase 7: Test Plan Generation (depends on all test tasks)
        test_plan_tasks = self._create_test_plan_tasks(
            spec, unit_test_tasks + integration_test_tasks + e2e_test_tasks
        )
        tasks.extend(test_plan_tasks)
        
        self.pipeline_state.tasks = tasks
        self._log_event("pipeline_built", {"task_count": len(tasks)})
        
        return self.pipeline_state
    
    def _create_database_tasks(self, spec: CanonicalSpec) -> List[Task]:
        """Create database schema generation tasks."""
        tasks = []
        
        # Convert user stories to dict format
        user_stories_data = [
            {
                'id': story.id,
                'title': story.title,
                'description': story.description,
                'acceptance_criteria': story.acceptance_criteria,
                'persona': story.persona,
                'priority': story.priority,
                'non_functional_hints': story.non_functional_hints,
                'tags': story.tags
            }
            for story in spec.user_stories
        ]
        
        # Schema generation task
        schema_task = Task(
            name="Generate Database Schema",
            description="Generate database schema and migrations based on user stories",
            agent_type=AgentType.DATABASE_CODER,
            input_data={
                "requirements": spec.requirements,
                "tech_stack": spec.tech_stack.get('database', 'PostgreSQL'),
                "user_stories": user_stories_data  # Add user stories
            }
        )
        tasks.append(schema_task)
        
        # ORM models task
        orm_task = Task(
            name="Generate ORM Models",
            description="Generate ORM models aligned with domain objects",
            agent_type=AgentType.DATABASE_CODER,
            dependencies=[schema_task.id],
            input_data={
                "schema_task_id": schema_task.id,
                "tech_stack": spec.tech_stack
            }
        )
        tasks.append(orm_task)
        
        return tasks
    
    def _create_backend_tasks(self, spec: CanonicalSpec, 
                             db_tasks: List[Task]) -> List[Task]:
        """Create backend code generation tasks."""
        tasks = []
        db_task_ids = [t.id for t in db_tasks]
        
        # Convert user stories to dict format
        user_stories_data = [
            {
                'id': story.id,
                'title': story.title,
                'description': story.description,
                'acceptance_criteria': story.acceptance_criteria,
                'persona': story.persona,
                'priority': story.priority,
                'non_functional_hints': story.non_functional_hints,
                'tags': story.tags
            }
            for story in spec.user_stories
        ]
        
        # API contracts task
        contracts_task = Task(
            name="Generate API Contracts",
            description="Generate REST API contracts and endpoints based on user stories",
            agent_type=AgentType.BACKEND_CODER,
            dependencies=db_task_ids,
            input_data={
                "requirements": spec.requirements,
                "tech_stack": spec.tech_stack.get('backend', 'FastAPI'),
                "user_stories": user_stories_data  # Add user stories
            }
        )
        tasks.append(contracts_task)
        
        # Services task
        services_task = Task(
            name="Generate Backend Services",
            description="Generate service layer with business logic",
            agent_type=AgentType.BACKEND_CODER,
            dependencies=[contracts_task.id],
            input_data={
                "contracts_task_id": contracts_task.id,
                "requirements": spec.requirements
            }
        )
        tasks.append(services_task)
        
        # Controllers task
        controllers_task = Task(
            name="Generate Controllers",
            description="Generate request handlers and validation logic",
            agent_type=AgentType.BACKEND_CODER,
            dependencies=[services_task.id],
            input_data={
                "services_task_id": services_task.id
            }
        )
        tasks.append(controllers_task)
        
        return tasks
    
    def _create_frontend_tasks(self, spec: CanonicalSpec,
                               backend_tasks: List[Task]) -> List[Task]:
        """Create frontend code generation tasks."""
        tasks = []
        # Frontend can start after API contracts are defined
        contracts_task_id = backend_tasks[0].id if backend_tasks else None
        
        # Convert user stories to dict format for JSON serialization
        user_stories_data = [
            {
                'id': story.id,
                'title': story.title,
                'description': story.description,
                'acceptance_criteria': story.acceptance_criteria,
                'persona': story.persona,
                'priority': story.priority,
                'non_functional_hints': story.non_functional_hints,
                'tags': story.tags
            }
            for story in spec.user_stories
        ]
        
        # Component scaffold task (can run parallel)
        scaffold_task = Task(
            name="Generate React Component Scaffold",
            description="Generate React component structure and routing based on user stories",
            agent_type=AgentType.FRONTEND_CODER,
            dependencies=[contracts_task_id] if contracts_task_id else [],
            input_data={
                "requirements": spec.requirements,
                "tech_stack": spec.tech_stack.get('frontend', 'React'),
                "user_stories": user_stories_data  # Add user stories
            }
        )
        tasks.append(scaffold_task)
        
        # Forms and state management task
        forms_task = Task(
            name="Generate Forms and State Management",
            description="Generate form components and state management",
            agent_type=AgentType.FRONTEND_CODER,
            dependencies=[scaffold_task.id],
            input_data={
                "scaffold_task_id": scaffold_task.id,
                "requirements": spec.requirements
            }
        )
        tasks.append(forms_task)
        
        # Accessibility checks task
        a11y_task = Task(
            name="Add Accessibility Features",
            description="Add accessibility checks and ARIA attributes",
            agent_type=AgentType.FRONTEND_CODER,
            dependencies=[forms_task.id],
            input_data={
                "forms_task_id": forms_task.id
            }
        )
        tasks.append(a11y_task)
        
        return tasks
    
    def _create_unit_test_tasks(self, spec: CanonicalSpec,
                               code_tasks: List[Task]) -> List[Task]:
        """Create unit test generation tasks."""
        tasks = []
        code_task_ids = [t.id for t in code_tasks]
        
        # Convert user stories to dict format
        user_stories_data = [
            {
                'id': story.id,
                'title': story.title,
                'description': story.description,
                'acceptance_criteria': story.acceptance_criteria,
                'persona': story.persona,
                'priority': story.priority,
                'non_functional_hints': story.non_functional_hints,
                'tags': story.tags
            }
            for story in spec.user_stories
        ]
        
        unit_test_task = Task(
            name="Generate Unit Tests",
            description="Generate unit tests from user stories and acceptance criteria",
            agent_type=AgentType.TESTING,
            dependencies=code_task_ids,
            input_data={
                "requirements": spec.requirements,
                "test_type": "unit",
                "coverage_threshold": self.coverage_threshold,
                "user_stories": user_stories_data  # Add user stories
            }
        )
        tasks.append(unit_test_task)
        
        return tasks
    
    def _create_integration_test_tasks(self, spec: CanonicalSpec,
                                       unit_test_tasks: List[Task]) -> List[Task]:
        """Create integration test generation tasks."""
        tasks = []
        unit_task_ids = [t.id for t in unit_test_tasks]
        
        # Convert user stories to dict format
        user_stories_data = [
            {
                'id': story.id,
                'title': story.title,
                'description': story.description,
                'acceptance_criteria': story.acceptance_criteria,
                'persona': story.persona,
                'priority': story.priority,
                'non_functional_hints': story.non_functional_hints,
                'tags': story.tags
            }
            for story in spec.user_stories
        ]
        
        integration_test_task = Task(
            name="Generate Integration Tests",
            description="Generate integration tests for API endpoints based on user stories",
            agent_type=AgentType.TESTING,
            dependencies=unit_task_ids,
            input_data={
                "requirements": spec.requirements,
                "test_type": "integration",
                "user_stories": user_stories_data  # Add user stories
            }
        )
        tasks.append(integration_test_task)
        
        return tasks
    
    def _create_e2e_test_tasks(self, spec: CanonicalSpec,
                              integration_tasks: List[Task]) -> List[Task]:
        """Create end-to-end test generation tasks."""
        tasks = []
        integration_task_ids = [t.id for t in integration_tasks]
        
        # Convert user stories to dict format
        user_stories_data = [
            {
                'id': story.id,
                'title': story.title,
                'description': story.description,
                'acceptance_criteria': story.acceptance_criteria,
                'persona': story.persona,
                'priority': story.priority,
                'non_functional_hints': story.non_functional_hints,
                'tags': story.tags
            }
            for story in spec.user_stories
        ]
        
        e2e_test_task = Task(
            name="Generate E2E Tests",
            description="Generate end-to-end tests with Playwright based on user stories",
            agent_type=AgentType.TESTING,
            dependencies=integration_task_ids,
            input_data={
                "requirements": spec.requirements,
                "test_type": "e2e",
                "user_stories": user_stories_data  # Add user stories
            }
        )
        tasks.append(e2e_test_task)
        
        return tasks
    
    def _create_test_plan_tasks(self, spec: CanonicalSpec,
                               test_tasks: List[Task]) -> List[Task]:
        """Create test plan generation tasks."""
        tasks = []
        test_task_ids = [t.id for t in test_tasks]
        
        test_plan_task = Task(
            name="Generate Test Plan",
            description="Generate comprehensive test plan for Azure DevOps",
            agent_type=AgentType.TESTING,
            dependencies=test_task_ids,
            input_data={
                "requirements": spec.requirements,
                "user_stories": [story.to_dict() for story in spec.user_stories],
                "project_name": spec.project_name or "Generated Code",
                "task_type": "test_plan"
            }
        )
        tasks.append(test_plan_task)
        
        return tasks
    
    def get_ready_tasks(self) -> List[Task]:
        """
        Get tasks that are ready to execute (all dependencies met).
        
        Returns:
            List of tasks ready for execution
        """
        if not self.pipeline_state:
            return []
        
        ready = []
        completed_ids = {
            t.id for t in self.pipeline_state.tasks 
            if t.status == TaskStatus.COMPLETED
        }
        
        for task in self.pipeline_state.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_met = all(dep_id in completed_ids for dep_id in task.dependencies)
            if deps_met:
                ready.append(task)
        
        # Limit to max parallel
        return ready[:self.max_parallel]
    
    def get_parallel_groups(self) -> List[List[Task]]:
        """
        Get groups of tasks that can run in parallel.
        
        Returns:
            List of task groups for parallel execution
        """
        if not self.pipeline_state:
            return []
        
        groups = []
        remaining = set(t.id for t in self.pipeline_state.tasks)
        completed: Set[str] = set()
        
        while remaining:
            # Find tasks whose dependencies are all completed
            ready_ids = set()
            for task in self.pipeline_state.tasks:
                if task.id not in remaining:
                    continue
                if all(dep_id in completed for dep_id in task.dependencies):
                    ready_ids.add(task.id)
            
            if not ready_ids:
                break  # Circular dependency or error
            
            # Create group
            group = [t for t in self.pipeline_state.tasks if t.id in ready_ids]
            groups.append(group)
            
            # Move to completed
            completed.update(ready_ids)
            remaining -= ready_ids
        
        return groups
    
    def update_task_status(self, task_id: str, status: TaskStatus,
                          output_data: Optional[Dict[str, Any]] = None,
                          error_message: Optional[str] = None) -> bool:
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task to update
            status: New status
            output_data: Output data from task execution
            error_message: Error message if failed
            
        Returns:
            True if update successful
        """
        if not self.pipeline_state:
            return False
        
        for task in self.pipeline_state.tasks:
            if task.id == task_id:
                task.status = status
                if output_data:
                    task.output_data = output_data
                if error_message:
                    task.error_message = error_message
                
                if status == TaskStatus.IN_PROGRESS:
                    task.started_at = datetime.now()
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    task.completed_at = datetime.now()
                
                self._log_event("task_status_changed", {
                    "task_id": task_id,
                    "status": status.value
                })
                
                self.pipeline_state.updated_at = datetime.now()
                return True
        
        return False
    
    def add_artifact(self, artifact: GeneratedArtifact) -> None:
        """Add a generated artifact to the pipeline state."""
        if self.pipeline_state:
            self.pipeline_state.artifacts.append(artifact)
            self._log_event("artifact_added", {
                "artifact_id": artifact.id,
                "file_path": artifact.file_path
            })
    
    def add_test_report(self, report: TestReport) -> None:
        """Add a test report to the pipeline state."""
        if self.pipeline_state:
            self.pipeline_state.test_reports.append(report)
            self._log_event("test_report_added", {
                "report_id": report.id,
                "passed": report.passed_tests,
                "failed": report.failed_tests
            })
    
    def check_quality_gate(self) -> Dict[str, Any]:
        """
        Check if quality gates are met.
        
        Returns:
            Dictionary with gate status and details
        """
        if not self.pipeline_state:
            return {"passed": False, "reason": "No pipeline state"}
        
        result = {
            "passed": True,
            "coverage_met": True,
            "tests_passed": True,
            "details": {}
        }
        
        # Check test results
        for report in self.pipeline_state.test_reports:
            if report.failed_tests > 0:
                result["passed"] = False
                result["tests_passed"] = False
                result["details"]["failed_tests"] = report.failed_tests
            
            if report.overall_coverage < self.coverage_threshold:
                result["passed"] = False
                result["coverage_met"] = False
                result["details"]["coverage"] = report.overall_coverage
                result["details"]["threshold"] = self.coverage_threshold
        
        self._log_event("quality_gate_check", result)
        return result
    
    def register_callback(self, callback: callable) -> None:
        """Register a callback for pipeline events."""
        self.event_callbacks.append(callback)
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a pipeline event."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.pipeline_state:
            self.pipeline_state.logs.append(event)
        
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors affect pipeline
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph as an adjacency list.
        
        Returns:
            Dictionary mapping task IDs to their dependent task IDs
        """
        if not self.pipeline_state:
            return {}
        
        graph = {}
        for task in self.pipeline_state.tasks:
            graph[task.id] = task.dependencies
        
        return graph
    
    def topological_sort(self) -> List[str]:
        """
        Get tasks in topological order (respecting dependencies).
        
        Returns:
            List of task IDs in execution order
        """
        if not self.pipeline_state:
            return []
        
        # Build in-degree map
        in_degree = {t.id: 0 for t in self.pipeline_state.tasks}
        for task in self.pipeline_state.tasks:
            for dep_id in task.dependencies:
                if dep_id in in_degree:
                    in_degree[task.id] += 1
        
        # Find tasks with no dependencies
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        result = []
        
        while queue:
            task_id = queue.popleft()
            result.append(task_id)
            
            # Reduce in-degree for dependent tasks
            for task in self.pipeline_state.tasks:
                if task_id in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)
        
        return result
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the current pipeline state."""
        if not self.pipeline_state:
            return {}
        
        status_counts = {}
        for task in self.pipeline_state.tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "pipeline_id": self.pipeline_state.id,
            "status": self.pipeline_state.status.value,
            "total_tasks": len(self.pipeline_state.tasks),
            "status_breakdown": status_counts,
            "artifacts_count": len(self.pipeline_state.artifacts),
            "test_reports_count": len(self.pipeline_state.test_reports),
            "created_at": self.pipeline_state.created_at.isoformat(),
            "updated_at": self.pipeline_state.updated_at.isoformat()
        }
