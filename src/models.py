"""
Data models for the agentic system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import uuid


class TaskStatus(Enum):
    """Status of a task in the pipeline"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class AgentType(Enum):
    """Types of agents in the system"""
    ADO_CONNECTOR = "ado_connector"
    ORCHESTRATOR = "orchestrator"
    FRONTEND_CODER = "frontend_coder"
    BACKEND_CODER = "backend_coder"
    DATABASE_CODER = "database_coder"
    TESTING = "testing"
    LEGACY_ANALYZER = "legacy_analyzer"
    PROMPT_REFINER = "prompt_refiner"
    MONITORING = "monitoring"


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"


@dataclass
class UserStory:
    """Represents a user story from ADO"""
    id: str
    title: str
    description: str
    acceptance_criteria: List[str] = field(default_factory=list)
    persona: Optional[str] = None
    priority: int = 3
    non_functional_hints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "persona": self.persona,
            "priority": self.priority,
            "non_functional_hints": self.non_functional_hints,
            "tags": self.tags
        }


@dataclass
class CanonicalSpec:
    """Normalized specification for downstream agents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_stories: List[UserStory] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    tech_stack: Dict[str, str] = field(default_factory=dict)
    project_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_stories": [s.to_dict() for s in self.user_stories],
            "requirements": self.requirements,
            "constraints": self.constraints,
            "tech_stack": self.tech_stack,
            "project_name": self.project_name,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Task:
    """Represents a task in the pipeline"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    agent_type: AgentType = AgentType.ORCHESTRATOR
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "artifacts": self.artifacts,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class TestResult:
    """Result of a test execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_name: str = ""
    test_type: TestType = TestType.UNIT
    passed: bool = False
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    coverage: Optional[float] = None
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "test_name": self.test_name,
            "test_type": self.test_type.value,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "coverage": self.coverage,
            "logs": self.logs
        }


@dataclass
class TestReport:
    """Complete test report"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    overall_coverage: float = 0.0
    duration_ms: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "overall_coverage": self.overall_coverage,
            "duration_ms": self.duration_ms,
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at.isoformat()
        }


@dataclass
class GeneratedArtifact:
    """Represents a generated code artifact"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    content: str = ""
    artifact_type: str = ""
    language: str = ""
    agent_type: AgentType = AgentType.ORCHESTRATOR
    documentation: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "content": self.content,
            "artifact_type": self.artifact_type,
            "language": self.language,
            "agent_type": self.agent_type.value,
            "documentation": self.documentation,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class LegacyAnalysis:
    """Analysis result from legacy repository"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repo_path: str = ""
    tech_stack: Dict[str, str] = field(default_factory=dict)
    architecture: str = ""
    dependencies: List[Dict[str, str]] = field(default_factory=list)
    conventions: List[str] = field(default_factory=list)
    integration_strategy: str = ""
    compatibility_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "repo_path": self.repo_path,
            "tech_stack": self.tech_stack,
            "architecture": self.architecture,
            "dependencies": self.dependencies,
            "conventions": self.conventions,
            "integration_strategy": self.integration_strategy,
            "compatibility_issues": self.compatibility_issues
        }


@dataclass
class TestCase:
    """Represents a test case in a test plan"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    steps: List[str] = field(default_factory=list)
    expected_result: str = ""
    test_type: TestType = TestType.UNIT
    priority: int = 2  # 1=Critical, 2=High, 3=Medium, 4=Low
    user_story_id: Optional[str] = None
    automated: bool = True
    automation_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": self.steps,
            "expected_result": self.expected_result,
            "test_type": self.test_type.value,
            "priority": self.priority,
            "user_story_id": self.user_story_id,
            "automated": self.automated,
            "automation_file": self.automation_file
        }


@dataclass
class TestSuite:
    """Represents a test suite containing multiple test cases"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    test_type: TestType = TestType.UNIT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "test_type": self.test_type.value
        }


@dataclass
class TestPlan:
    """Represents an Azure DevOps test plan"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_suites: List[TestSuite] = field(default_factory=list)
    project: str = ""
    iteration: Optional[str] = None
    area_path: Optional[str] = None
    ado_id: Optional[int] = None  # Azure DevOps test plan ID after creation
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "test_suites": [ts.to_dict() for ts in self.test_suites],
            "project": self.project,
            "iteration": self.iteration,
            "area_path": self.area_path,
            "ado_id": self.ado_id,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class PipelineState:
    """Current state of the entire pipeline"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    tasks: List[Task] = field(default_factory=list)
    artifacts: List[GeneratedArtifact] = field(default_factory=list)
    test_reports: List[TestReport] = field(default_factory=list)
    test_plans: List[TestPlan] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    spec: Optional[CanonicalSpec] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    commit_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "tasks": [t.to_dict() for t in self.tasks],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "test_reports": [r.to_dict() for r in self.test_reports],
            "test_plans": [tp.to_dict() for tp in self.test_plans],
            "logs": self.logs,
            "spec": self.spec.to_dict() if self.spec else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "commit_result": self.commit_result
        }
