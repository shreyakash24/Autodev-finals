"""
Testing Agent
Executes tests in isolated runners (Pytest for Python, Playwright for E2E).
Produces compact pass/fail reports with logs, coverage summaries, and failing traces.
"""

import subprocess
import json
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from crewai import Agent

from ..models import (
    TestResult, TestReport, TestType, AgentType,
    GeneratedArtifact, UserStory, TestPlan, TestSuite, TestCase
)
from ..utils.llm_config import get_llm

# Constants
MAX_TEST_CASES_PER_STORY = 3  # Limit test cases per story to avoid overwhelming test plans
INVALID_PATH_CHARS = ['<', '>', ':', '"', '|', '?', '*']  # Azure DevOps restricted characters
MAX_WORK_ITEMS_PER_COMPONENT = 5  # Limit feature work items to prevent overwhelming boards


class TestingAgent:
    """
    Testing Agent that executes and generates tests.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Testing Agent                                    │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Test Generator                                 │ │
    │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │ │
    │  │  │ Unit Tests  │  │ Integration  │  │ E2E Tests (Playwright)   │  │ │
    │  │  └─────────────┘  └──────────────┘  └───────────────────────────┘  │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Test Runner                                    │ │
    │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │ │
    │  │  │  Pytest     │  │ Coverage     │  │ Report Generator          │  │ │
    │  │  └─────────────┘  └──────────────┘  └───────────────────────────┘  │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, coverage_threshold: float = 80.0):
        """
        Initialize the Testing Agent.
        
        Args:
            coverage_threshold: Minimum coverage percentage required
        """
        self.coverage_threshold = coverage_threshold
        self.agent_type = AgentType.TESTING
        
        self.llm = get_llm(temperature=0.1)
        
        self.crew_agent = Agent(
            role="QA Engineer",
            goal="Generate comprehensive tests and execute them reliably",
            backstory="""You are an expert QA engineer who creates thorough
            test suites covering edge cases and ensures high code coverage.
            You use pytest for Python tests and Playwright for E2E tests.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        ) if self.llm else None
    
    def generate_unit_tests(self, requirements: Dict[str, Any],
                           artifacts: List[GeneratedArtifact]) -> List[GeneratedArtifact]:
        """
        Generate unit tests from acceptance criteria.
        
        Args:
            requirements: Functional requirements
            artifacts: Code artifacts to test
            
        Returns:
            List of test file artifacts
        """
        # If LLM is available and we have user stories, use LLM-based generation
        if self.llm and self.crew_agent and requirements.get('user_stories'):
            return self._generate_unit_tests_from_stories(requirements, artifacts)
        
        # Fallback to template-based generation
        return self._generate_unit_tests_from_templates(requirements, artifacts)
    
    def _generate_unit_tests_from_stories(self, requirements: Dict[str, Any], 
                                          artifacts: List[GeneratedArtifact]) -> List[GeneratedArtifact]:
        """Generate unit tests using LLM based on user stories and artifacts."""
        test_artifacts = []
        user_stories = requirements.get('user_stories', [])
        
        if not user_stories:
            return self._generate_unit_tests_from_templates(requirements, artifacts)
        
        from crewai import Task as CrewTask
        
        # Build context from user stories and artifacts
        stories_text = "\n".join([
            f"Story {i+1}: {story.get('title', 'Untitled')}\n"
            f"Description: {story.get('description', 'No description')}\n"
            f"Acceptance Criteria: {', '.join(story.get('acceptance_criteria', []))}\n"
            for i, story in enumerate(user_stories)
        ])
        
        # Get artifact file paths and types
        artifact_info = "\n".join([
            f"- {art.file_path} ({art.artifact_type}, {art.language})"
            for art in artifacts[:10]  # Limit to first 10 to avoid token limits
        ])
        
        # Generate pytest unit tests
        unit_test_task = CrewTask(
            description=f"""Generate comprehensive pytest unit tests for these user stories:
{stories_text}

Generated artifacts to test:
{artifact_info}

Create unit tests with:
1. Test functions for each acceptance criterion
2. Fixtures for test data
3. Mocks for external dependencies
4. Assertions for expected behavior
5. Edge case and error handling tests
6. Proper test organization and naming

Return ONLY the complete Python pytest code for tests/test_services.py, no explanations.""",
            agent=self.crew_agent,
            expected_output="Complete Python pytest unit test code"
        )
        
        try:
            test_result = unit_test_task.execute_sync()
            test_content = str(test_result) if test_result else self._generate_service_tests(requirements)
            
            test_artifacts.append(GeneratedArtifact(
                file_path="tests/test_services.py",
                content=test_content,
                artifact_type="test",
                language="python",
                agent_type=self.agent_type,
                documentation="Unit tests for services"
            ))
        except Exception as e:
            print(f"Error generating unit tests with LLM: {e}")
            # Fallback to template
            return self._generate_unit_tests_from_templates(requirements, artifacts)
        
        return test_artifacts
    
    def _generate_unit_tests_from_templates(self, requirements: Dict[str, Any],
                                           artifacts: List[GeneratedArtifact]) -> List[GeneratedArtifact]:
        """Generate unit tests from templates (fallback)."""
        test_artifacts = []
        
        # Generate tests for backend services
        service_tests = self._generate_service_tests(requirements)
        test_artifacts.append(service_tests)
        
        # Generate tests for API routes
        api_tests = self._generate_api_tests(requirements)
        test_artifacts.append(api_tests)
        
        # Generate tests for schemas
        schema_tests = self._generate_schema_tests(requirements)
        test_artifacts.append(schema_tests)
        
        return test_artifacts
    
    def generate_integration_tests(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """
        Generate integration tests for API endpoints.
        
        Args:
            requirements: Functional requirements
            
        Returns:
            List of test file artifacts
        """
        # If LLM is available and we have user stories, use LLM-based generation
        if self.llm and self.crew_agent and requirements.get('user_stories'):
            return self._generate_integration_tests_from_stories(requirements)
        
        # Fallback to template-based generation
        return self._generate_integration_tests_from_templates(requirements)
    
    def _generate_integration_tests_from_stories(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """Generate integration tests using LLM based on user stories."""
        test_artifacts = []
        user_stories = requirements.get('user_stories', [])
        
        if not user_stories:
            return self._generate_integration_tests_from_templates(requirements)
        
        from crewai import Task as CrewTask
        
        stories_text = "\n".join([
            f"Story {i+1}: {story.get('title', 'Untitled')}\n"
            f"Description: {story.get('description', 'No description')}\n"
            f"Acceptance Criteria: {', '.join(story.get('acceptance_criteria', []))}\n"
            for i, story in enumerate(user_stories)
        ])
        
        integration_test_task = CrewTask(
            description=f"""Generate comprehensive pytest integration tests for these user stories:
{stories_text}

Create integration tests with:
1. Test functions for API endpoints
2. Database transaction tests
3. Authentication/authorization tests
4. Request/response validation
5. Error handling and status codes
6. Test fixtures for test database setup

Return ONLY the complete Python pytest code for tests/test_integration.py, no explanations.""",
            agent=self.crew_agent,
            expected_output="Complete Python pytest integration test code"
        )
        
        try:
            test_result = integration_test_task.execute_sync()
            test_content = str(test_result) if test_result else self._generate_integration_test_file(requirements)
            
            test_artifacts.append(GeneratedArtifact(
                file_path="tests/test_integration.py",
                content=test_content,
                artifact_type="test",
                language="python",
                agent_type=self.agent_type,
                documentation="Integration tests for API endpoints"
            ))
        except Exception as e:
            print(f"Error generating integration tests with LLM: {e}")
            return self._generate_integration_tests_from_templates(requirements)
        
        return test_artifacts
    
    def _generate_integration_tests_from_templates(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """Generate integration tests from templates (fallback)."""
        test_artifacts = []
        integration_tests = self._generate_integration_test_file(requirements)
        test_artifacts.append(integration_tests)
        return test_artifacts
    
    def generate_e2e_tests(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """
        Generate end-to-end tests with Playwright.
        
        Args:
            requirements: Functional requirements
            
        Returns:
            List of test file artifacts
        """
        # If LLM is available and we have user stories, use LLM-based generation
        if self.llm and self.crew_agent and requirements.get('user_stories'):
            return self._generate_e2e_tests_from_stories(requirements)
        
        # Fallback to template-based generation
        return self._generate_e2e_tests_from_templates(requirements)
    
    def _generate_e2e_tests_from_stories(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """Generate E2E tests using LLM based on user stories."""
        test_artifacts = []
        user_stories = requirements.get('user_stories', [])
        
        if not user_stories:
            return self._generate_e2e_tests_from_templates(requirements)
        
        from crewai import Task as CrewTask
        
        stories_text = "\n".join([
            f"Story {i+1}: {story.get('title', 'Untitled')}\n"
            f"Description: {story.get('description', 'No description')}\n"
            f"Acceptance Criteria: {', '.join(story.get('acceptance_criteria', []))}\n"
            for i, story in enumerate(user_stories)
        ])
        
        e2e_test_task = CrewTask(
            description=f"""Generate comprehensive Playwright E2E tests for these user stories:
{stories_text}

Create E2E tests with:
1. Test functions for user workflows
2. Page object patterns
3. Element selectors and interactions
4. Assertions for UI state
5. Screenshot capture on failure
6. Proper test isolation and cleanup

Return ONLY the complete TypeScript/JavaScript Playwright code for tests/e2e/test_flows.spec.ts, no explanations.""",
            agent=self.crew_agent,
            expected_output="Complete Playwright E2E test code"
        )
        
        try:
            test_result = e2e_test_task.execute_sync()
            test_content = str(test_result) if test_result else self._generate_playwright_tests(requirements)
            
            test_artifacts.append(GeneratedArtifact(
                file_path="tests/e2e/test_flows.spec.ts",
                content=test_content,
                artifact_type="test",
                language="typescript",
                agent_type=self.agent_type,
                documentation="End-to-end tests with Playwright"
            ))
        except Exception as e:
            print(f"Error generating E2E tests with LLM: {e}")
            return self._generate_e2e_tests_from_templates(requirements)
        
        # Always generate config (not story-specific)
        config = self._generate_playwright_config()
        test_artifacts.append(config)
        
        return test_artifacts
    
    def _generate_e2e_tests_from_templates(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """Generate E2E tests from templates (fallback)."""
        test_artifacts = []
        
        # Playwright test file
        e2e_tests = self._generate_playwright_tests(requirements)
        test_artifacts.append(e2e_tests)
        
        # Playwright config
        config = self._generate_playwright_config()
        test_artifacts.append(config)
        
        return test_artifacts
    
    def run_tests(self, test_path: str, test_type: TestType = TestType.UNIT,
                 collect_coverage: bool = True) -> TestReport:
        """
        Execute tests and collect results.
        
        Args:
            test_path: Path to tests to run
            test_type: Type of tests being run
            collect_coverage: Whether to collect coverage
            
        Returns:
            TestReport with results
        """
        start_time = datetime.now()
        results: List[TestResult] = []
        
        try:
            if test_type == TestType.E2E:
                results, coverage = self._run_playwright_tests(test_path)
            else:
                results, coverage = self._run_pytest(test_path, collect_coverage)
        except Exception as e:
            # Create a failed result if execution fails
            results = [TestResult(
                test_name="test_execution",
                test_type=test_type,
                passed=False,
                error_message=str(e),
                logs=[f"Test execution failed: {e}"]
            )]
            coverage = 0.0
        
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build report
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        
        return TestReport(
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=0,
            overall_coverage=coverage,
            duration_ms=duration,
            results=results
        )
    
    def _run_pytest(self, test_path: str, 
                   collect_coverage: bool = True) -> Tuple[List[TestResult], float]:
        """
        Run pytest and collect results.
        
        Args:
            test_path: Path to tests
            collect_coverage: Whether to collect coverage
            
        Returns:
            Tuple of (results, coverage_percentage)
        """
        results = []
        coverage = 0.0
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", test_path, "-v", "--json-report"]
        
        if collect_coverage:
            cmd.extend(["--cov=app", "--cov-report=json"])
        
        try:
            # Run pytest
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse JSON report if available
            json_report_path = ".report.json"
            if os.path.exists(json_report_path):
                with open(json_report_path, 'r') as f:
                    report_data = json.load(f)
                
                for test in report_data.get('tests', []):
                    results.append(TestResult(
                        test_name=test.get('nodeid', 'unknown'),
                        test_type=TestType.UNIT,
                        passed=test.get('outcome') == 'passed',
                        duration_ms=test.get('duration', 0) * 1000,
                        error_message=test.get('call', {}).get('longrepr') if test.get('outcome') != 'passed' else None,
                        logs=[process.stdout] if process.stdout else []
                    ))
            
            # Parse coverage report
            coverage_path = "coverage.json"
            if collect_coverage and os.path.exists(coverage_path):
                with open(coverage_path, 'r') as f:
                    cov_data = json.load(f)
                coverage = cov_data.get('totals', {}).get('percent_covered', 0.0)
                
        except subprocess.TimeoutExpired:
            results.append(TestResult(
                test_name="pytest_execution",
                test_type=TestType.UNIT,
                passed=False,
                error_message="Test execution timed out",
                logs=["Timeout after 300 seconds"]
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="pytest_execution",
                test_type=TestType.UNIT,
                passed=False,
                error_message=str(e),
                logs=[str(e)]
            ))
        
        return results, coverage
    
    def _run_playwright_tests(self, test_path: str) -> Tuple[List[TestResult], float]:
        """
        Run Playwright E2E tests.
        
        Args:
            test_path: Path to test files
            
        Returns:
            Tuple of (results, coverage - always 0 for E2E)
        """
        results = []
        
        cmd = ["npx", "playwright", "test", test_path, "--reporter=json"]
        
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Parse Playwright JSON output
            try:
                output_data = json.loads(process.stdout)
                for suite in output_data.get('suites', []):
                    for spec in suite.get('specs', []):
                        for test in spec.get('tests', []):
                            results.append(TestResult(
                                test_name=f"{spec.get('title', '')} > {test.get('title', '')}",
                                test_type=TestType.E2E,
                                passed=test.get('status') == 'expected',
                                duration_ms=test.get('results', [{}])[0].get('duration', 0),
                                error_message=test.get('results', [{}])[0].get('error', {}).get('message'),
                                stack_trace=test.get('results', [{}])[0].get('error', {}).get('stack'),
                                logs=[]
                            ))
            except json.JSONDecodeError:
                # If JSON parsing fails, create result from stdout
                passed = process.returncode == 0
                results.append(TestResult(
                    test_name="playwright_suite",
                    test_type=TestType.E2E,
                    passed=passed,
                    error_message=process.stderr if not passed else None,
                    logs=[process.stdout, process.stderr]
                ))
                
        except subprocess.TimeoutExpired:
            results.append(TestResult(
                test_name="playwright_execution",
                test_type=TestType.E2E,
                passed=False,
                error_message="E2E test execution timed out",
                logs=["Timeout after 600 seconds"]
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="playwright_execution",
                test_type=TestType.E2E,
                passed=False,
                error_message=str(e),
                logs=[str(e)]
            ))
        
        return results, 0.0  # E2E tests don't have traditional coverage
    
    def generate_report(self, report: TestReport) -> str:
        """
        Generate a compact pass/fail report.
        
        Args:
            report: TestReport object
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "TEST EXECUTION REPORT",
            "=" * 60,
            f"",
            f"Total Tests: {report.total_tests}",
            f"✅ Passed: {report.passed_tests}",
            f"❌ Failed: {report.failed_tests}",
            f"⏭️ Skipped: {report.skipped_tests}",
            f"",
            f"Coverage: {report.overall_coverage:.1f}%",
            f"Duration: {report.duration_ms:.0f}ms",
            f"",
        ]
        
        # Add coverage status
        if report.overall_coverage >= self.coverage_threshold:
            lines.append(f"✅ Coverage threshold met ({self.coverage_threshold}%)")
        else:
            lines.append(f"❌ Coverage below threshold ({self.coverage_threshold}%)")
        
        lines.append("")
        
        # Add failed test details
        failed_tests = [r for r in report.results if not r.passed]
        if failed_tests:
            lines.append("-" * 60)
            lines.append("FAILED TESTS:")
            lines.append("-" * 60)
            
            for test in failed_tests:
                lines.append(f"")
                lines.append(f"❌ {test.test_name}")
                if test.error_message:
                    lines.append(f"   Error: {test.error_message[:200]}")
                if test.stack_trace:
                    # Show first 5 lines of stack trace
                    trace_lines = test.stack_trace.split('\n')[:5]
                    for trace_line in trace_lines:
                        lines.append(f"   {trace_line}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _generate_service_tests(self, requirements: Dict[str, Any]) -> GeneratedArtifact:
        """Generate unit tests for service layer."""
        content = '''"""
Unit Tests for Service Layer

Architecture:
Tests cover CRUD operations, edge cases, and error handling.
Uses pytest fixtures for setup and teardown.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from app.services.base import BaseService
from app.schemas import ItemCreate, ItemUpdate, ItemResponse


@pytest.fixture
def service():
    """Create a fresh service instance for each test."""
    return BaseService()


@pytest.fixture
def sample_item_data():
    """Sample item data for testing."""
    return {
        "name": "Test Item",
        "description": "A test item description",
        "status": "active"
    }


class TestBaseService:
    """Test suite for BaseService."""
    
    @pytest.mark.asyncio
    async def test_create_item_success(self, service, sample_item_data):
        """Test successful item creation."""
        item_create = ItemCreate(**sample_item_data)
        result = await service.create_item(item_create)
        
        assert result is not None
        assert result.name == sample_item_data["name"]
        assert result.description == sample_item_data["description"]
        assert result.status == sample_item_data["status"]
        assert result.id is not None
        assert result.created_at is not None
    
    @pytest.mark.asyncio
    async def test_get_item_success(self, service, sample_item_data):
        """Test getting an existing item."""
        # Create item first
        item_create = ItemCreate(**sample_item_data)
        created = await service.create_item(item_create)
        
        # Get the item
        result = await service.get_item(created.id)
        
        assert result is not None
        assert result.id == created.id
        assert result.name == sample_item_data["name"]
    
    @pytest.mark.asyncio
    async def test_get_item_not_found(self, service):
        """Test getting a non-existent item."""
        result = await service.get_item("non-existent-id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_items_empty(self, service):
        """Test listing items when none exist."""
        items, total = await service.list_items()
        
        assert items == []
        assert total == 0
    
    @pytest.mark.asyncio
    async def test_list_items_with_pagination(self, service, sample_item_data):
        """Test listing items with pagination."""
        # Create multiple items
        for i in range(15):
            item_create = ItemCreate(
                name=f"Item {i}",
                description=f"Description {i}",
                status="active"
            )
            await service.create_item(item_create)
        
        # Test first page
        items, total = await service.list_items(skip=0, limit=10)
        assert len(items) == 10
        assert total == 15
        
        # Test second page
        items, total = await service.list_items(skip=10, limit=10)
        assert len(items) == 5
        assert total == 15
    
    @pytest.mark.asyncio
    async def test_list_items_with_search(self, service):
        """Test searching items."""
        # Create items with different names
        await service.create_item(ItemCreate(name="Apple", description="Fruit"))
        await service.create_item(ItemCreate(name="Banana", description="Yellow fruit"))
        await service.create_item(ItemCreate(name="Carrot", description="Vegetable"))
        
        # Search for "fruit"
        items, total = await service.list_items(search="fruit")
        assert total == 2
    
    @pytest.mark.asyncio
    async def test_update_item_success(self, service, sample_item_data):
        """Test updating an item."""
        # Create item
        item_create = ItemCreate(**sample_item_data)
        created = await service.create_item(item_create)
        
        # Update item
        update_data = ItemUpdate(name="Updated Name")
        result = await service.update_item(created.id, update_data)
        
        assert result is not None
        assert result.name == "Updated Name"
        assert result.description == sample_item_data["description"]
    
    @pytest.mark.asyncio
    async def test_update_item_not_found(self, service):
        """Test updating a non-existent item."""
        update_data = ItemUpdate(name="Updated Name")
        result = await service.update_item("non-existent-id", update_data)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_item_success(self, service, sample_item_data):
        """Test deleting an item."""
        # Create item
        item_create = ItemCreate(**sample_item_data)
        created = await service.create_item(item_create)
        
        # Delete item
        result = await service.delete_item(created.id)
        assert result is True
        
        # Verify deletion
        item = await service.get_item(created.id)
        assert item is None
    
    @pytest.mark.asyncio
    async def test_delete_item_not_found(self, service):
        """Test deleting a non-existent item."""
        result = await service.delete_item("non-existent-id")
        assert result is False
'''
        return GeneratedArtifact(
            file_path="tests/test_services.py",
            content=content,
            artifact_type="test",
            language="python",
            agent_type=self.agent_type,
            documentation="Unit tests for service layer"
        )
    
    def _generate_api_tests(self, requirements: Dict[str, Any]) -> GeneratedArtifact:
        """Generate API route tests."""
        content = '''"""
API Route Tests

Architecture:
Tests API endpoints using FastAPI TestClient.
Covers request/response validation, error handling, and edge cases.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestItemsAPI:
    """Test items CRUD API."""
    
    def test_create_item(self, client):
        """Test creating a new item."""
        item_data = {
            "name": "Test Item",
            "description": "Test description",
            "status": "active"
        }
        response = client.post("/api/v1/items", json=item_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == item_data["name"]
        assert "id" in data
    
    def test_create_item_validation_error(self, client):
        """Test creating item with invalid data."""
        invalid_data = {"name": ""}  # Empty name should fail
        response = client.post("/api/v1/items", json=invalid_data)
        
        assert response.status_code == 422
    
    def test_list_items(self, client):
        """Test listing items."""
        response = client.get("/api/v1/items")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    def test_list_items_pagination(self, client):
        """Test pagination parameters."""
        response = client.get("/api/v1/items?skip=0&limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert data["skip"] == 0
        assert data["limit"] == 5
    
    def test_get_item_not_found(self, client):
        """Test getting non-existent item."""
        response = client.get("/api/v1/items/non-existent-id")
        
        assert response.status_code == 404
    
    def test_update_item(self, client):
        """Test updating an item."""
        # First create an item
        create_response = client.post("/api/v1/items", json={
            "name": "Original Name",
            "description": "Original description"
        })
        item_id = create_response.json()["id"]
        
        # Update it
        update_response = client.put(f"/api/v1/items/{item_id}", json={
            "name": "Updated Name"
        })
        
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Updated Name"
    
    def test_delete_item(self, client):
        """Test deleting an item."""
        # First create an item
        create_response = client.post("/api/v1/items", json={
            "name": "To Delete"
        })
        item_id = create_response.json()["id"]
        
        # Delete it
        delete_response = client.delete(f"/api/v1/items/{item_id}")
        
        assert delete_response.status_code == 204
        
        # Verify it's gone
        get_response = client.get(f"/api/v1/items/{item_id}")
        assert get_response.status_code == 404
'''
        return GeneratedArtifact(
            file_path="tests/test_api.py",
            content=content,
            artifact_type="test",
            language="python",
            agent_type=self.agent_type,
            documentation="API endpoint tests"
        )
    
    def _generate_schema_tests(self, requirements: Dict[str, Any]) -> GeneratedArtifact:
        """Generate schema validation tests."""
        content = '''"""
Schema Validation Tests

Architecture:
Tests Pydantic schema validation rules.
"""

import pytest
from pydantic import ValidationError
from app.schemas import ItemCreate, ItemUpdate, ItemResponse


class TestItemSchemas:
    """Test item schemas."""
    
    def test_item_create_valid(self):
        """Test valid item creation schema."""
        item = ItemCreate(
            name="Valid Name",
            description="Valid description",
            status="active"
        )
        assert item.name == "Valid Name"
    
    def test_item_create_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValidationError):
            ItemCreate(description="No name")
    
    def test_item_create_name_min_length(self):
        """Test name minimum length."""
        with pytest.raises(ValidationError):
            ItemCreate(name="")  # Empty name
    
    def test_item_create_name_max_length(self):
        """Test name maximum length."""
        with pytest.raises(ValidationError):
            ItemCreate(name="x" * 101)  # Over 100 chars
    
    def test_item_update_all_optional(self):
        """Test that all update fields are optional."""
        item = ItemUpdate()  # No fields provided
        assert item.name is None
        assert item.description is None
    
    def test_item_update_partial(self):
        """Test partial update."""
        item = ItemUpdate(name="New Name")
        assert item.name == "New Name"
        assert item.description is None
'''
        return GeneratedArtifact(
            file_path="tests/test_schemas.py",
            content=content,
            artifact_type="test",
            language="python",
            agent_type=self.agent_type,
            documentation="Schema validation tests"
        )
    
    def _generate_integration_test_file(self, requirements: Dict[str, Any]) -> GeneratedArtifact:
        """Generate integration tests."""
        content = '''"""
Integration Tests

Architecture:
End-to-end tests for complete workflows.
Tests interactions between components.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestItemWorkflow:
    """Test complete item workflow."""
    
    def test_complete_item_lifecycle(self, client):
        """Test create -> read -> update -> delete workflow."""
        # Create
        create_data = {"name": "Lifecycle Test", "status": "active"}
        create_response = client.post("/api/v1/items", json=create_data)
        assert create_response.status_code == 201
        item_id = create_response.json()["id"]
        
        # Read
        get_response = client.get(f"/api/v1/items/{item_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "Lifecycle Test"
        
        # Update
        update_response = client.put(
            f"/api/v1/items/{item_id}",
            json={"name": "Updated Lifecycle Test"}
        )
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Updated Lifecycle Test"
        
        # Delete
        delete_response = client.delete(f"/api/v1/items/{item_id}")
        assert delete_response.status_code == 204
        
        # Verify deleted
        verify_response = client.get(f"/api/v1/items/{item_id}")
        assert verify_response.status_code == 404
    
    def test_bulk_operations(self, client):
        """Test creating and listing multiple items."""
        # Create multiple items
        items_to_create = [
            {"name": f"Bulk Item {i}", "status": "active"}
            for i in range(5)
        ]
        
        created_ids = []
        for item_data in items_to_create:
            response = client.post("/api/v1/items", json=item_data)
            assert response.status_code == 201
            created_ids.append(response.json()["id"])
        
        # List all items
        list_response = client.get("/api/v1/items?limit=100")
        assert list_response.status_code == 200
        
        # Verify all items are present
        items = list_response.json()["items"]
        item_ids = [item["id"] for item in items]
        for created_id in created_ids:
            assert created_id in item_ids
        
        # Cleanup
        for item_id in created_ids:
            client.delete(f"/api/v1/items/{item_id}")
    
    def test_search_integration(self, client):
        """Test search functionality integration."""
        # Create items with searchable content
        client.post("/api/v1/items", json={"name": "Apple Pie", "description": "Dessert"})
        client.post("/api/v1/items", json={"name": "Banana Bread", "description": "Baked good"})
        client.post("/api/v1/items", json={"name": "Carrot Cake", "description": "Dessert"})
        
        # Search by name
        search_response = client.get("/api/v1/items?search=apple")
        assert search_response.status_code == 200
        items = search_response.json()["items"]
        assert any("Apple" in item["name"] for item in items)
'''
        return GeneratedArtifact(
            file_path="tests/test_integration.py",
            content=content,
            artifact_type="test",
            language="python",
            agent_type=self.agent_type,
            documentation="Integration tests"
        )
    
    def _generate_playwright_tests(self, requirements: Dict[str, Any]) -> GeneratedArtifact:
        """Generate Playwright E2E tests."""
        content = '''/**
 * End-to-End Tests with Playwright
 * 
 * Architecture:
 * Tests complete user workflows through the UI.
 */

import { test, expect } from '@playwright/test';

test.describe('Application E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the home page', async ({ page }) => {
    await expect(page).toHaveTitle(/Generated App/);
    await expect(page.locator('main')).toBeVisible();
  });

  test('should navigate to items page', async ({ page }) => {
    await page.click('a[href="/items"]');
    await expect(page).toHaveURL(/.*items/);
  });

  test('should create a new item', async ({ page }) => {
    // Navigate to create form
    await page.click('a[href="/items/new"]');
    
    // Fill form
    await page.fill('input[name="name"]', 'E2E Test Item');
    await page.fill('textarea[name="description"]', 'Created by E2E test');
    
    // Submit
    await page.click('button[type="submit"]');
    
    // Verify success
    await expect(page.locator('.success-message')).toBeVisible();
  });

  test('should display validation errors', async ({ page }) => {
    await page.click('a[href="/items/new"]');
    
    // Submit empty form
    await page.click('button[type="submit"]');
    
    // Check for validation error
    await expect(page.locator('.field-error')).toBeVisible();
  });

  test('should be accessible', async ({ page }) => {
    // Check skip link
    await page.keyboard.press('Tab');
    const skipLink = page.locator('.skip-link');
    await expect(skipLink).toBeFocused();
    
    // Check main content has proper landmark
    await expect(page.locator('main')).toHaveAttribute('id', 'main-content');
  });

  test('should handle errors gracefully', async ({ page }) => {
    // Try to access non-existent item
    await page.goto('/items/non-existent-id');
    
    // Should show error message
    await expect(page.locator('[role="alert"]')).toBeVisible();
  });
});

test.describe('Form Interactions', () => {
  test('should validate required fields', async ({ page }) => {
    await page.goto('/items/new');
    
    // Focus and blur required field without entering data
    await page.focus('input[name="name"]');
    await page.blur('input[name="name"]');
    
    // Check for error
    await expect(page.locator('#name-error')).toBeVisible();
  });

  test('should clear errors on input', async ({ page }) => {
    await page.goto('/items/new');
    
    // Trigger error
    await page.focus('input[name="name"]');
    await page.blur('input[name="name"]');
    await expect(page.locator('#name-error')).toBeVisible();
    
    // Enter valid data
    await page.fill('input[name="name"]', 'Valid Name');
    
    // Error should be cleared
    await expect(page.locator('#name-error')).not.toBeVisible();
  });
});
'''
        return GeneratedArtifact(
            file_path="tests/e2e/app.spec.ts",
            content=content,
            artifact_type="test",
            language="typescript",
            agent_type=self.agent_type,
            documentation="Playwright E2E tests"
        )
    
    def _generate_playwright_config(self) -> GeneratedArtifact:
        """Generate Playwright configuration."""
        content = '''/**
 * Playwright Configuration
 * 
 * Architecture:
 * Configures test runners, browsers, and reporting.
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/results.json' }],
  ],
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
'''
        return GeneratedArtifact(
            file_path="playwright.config.ts",
            content=content,
            artifact_type="config",
            language="typescript",
            agent_type=self.agent_type,
            documentation="Playwright test configuration"
        )
    
    def generate_test_plan(self, user_stories: List[UserStory], 
                          artifacts: List[GeneratedArtifact],
                          project_name: str) -> TestPlan:
        """
        Generate a comprehensive test plan from user stories and generated artifacts.
        
        Args:
            user_stories: List of user stories with acceptance criteria
            artifacts: Generated code artifacts
            project_name: Name of the project
            
        Returns:
            TestPlan with test suites and test cases
        """
        print(f"[Testing Agent] Generating test plan for {len(user_stories)} user stories")
        
        test_plan = TestPlan(
            name=f"{project_name} Test Plan",
            description=f"Comprehensive test plan for {project_name} project",
            project=project_name
        )
        
        # Create test suites for different test types
        unit_suite = TestSuite(
            name="Unit Tests",
            description="Unit tests for individual components and services",
            test_type=TestType.UNIT
        )
        
        integration_suite = TestSuite(
            name="Integration Tests",
            description="Integration tests for API endpoints and data flow",
            test_type=TestType.INTEGRATION
        )
        
        e2e_suite = TestSuite(
            name="End-to-End Tests",
            description="End-to-end tests for complete user workflows",
            test_type=TestType.E2E
        )
        
        # Generate test cases for each user story
        for story in user_stories:
            # Create unit test cases
            unit_tests = self._generate_test_cases_for_story(
                story, TestType.UNIT, artifacts
            )
            unit_suite.test_cases.extend(unit_tests)
            
            # Create integration test cases
            integration_tests = self._generate_test_cases_for_story(
                story, TestType.INTEGRATION, artifacts
            )
            integration_suite.test_cases.extend(integration_tests)
            
            # Create E2E test cases based on acceptance criteria
            e2e_tests = self._generate_e2e_test_cases_for_story(story)
            e2e_suite.test_cases.extend(e2e_tests)
        
        # Add suites to test plan
        if unit_suite.test_cases:
            test_plan.test_suites.append(unit_suite)
            print(f"[Testing Agent] Created unit test suite with {len(unit_suite.test_cases)} test cases")
        
        if integration_suite.test_cases:
            test_plan.test_suites.append(integration_suite)
            print(f"[Testing Agent] Created integration test suite with {len(integration_suite.test_cases)} test cases")
        
        if e2e_suite.test_cases:
            test_plan.test_suites.append(e2e_suite)
            print(f"[Testing Agent] Created E2E test suite with {len(e2e_suite.test_cases)} test cases")
        
        return test_plan
    
    def _generate_test_cases_for_story(self, story: UserStory, 
                                      test_type: TestType,
                                      artifacts: List[GeneratedArtifact]) -> List[TestCase]:
        """
        Generate test cases for a user story based on test type.
        
        Args:
            story: User story with acceptance criteria
            test_type: Type of tests to generate
            artifacts: Code artifacts that might be tested
            
        Returns:
            List of test cases
        """
        test_cases = []
        
        # Find related test files
        automation_file = None
        if test_type == TestType.UNIT:
            # Look for unit test files
            for artifact in artifacts:
                if 'test_' in artifact.file_path and 'service' in artifact.file_path:
                    automation_file = artifact.file_path
                    break
        elif test_type == TestType.INTEGRATION:
            # Look for API test files
            for artifact in artifacts:
                if 'test_' in artifact.file_path and 'api' in artifact.file_path:
                    automation_file = artifact.file_path
                    break
        
        # Generate test case for each acceptance criterion (limited)
        for i, criterion in enumerate(story.acceptance_criteria[:MAX_TEST_CASES_PER_STORY], 1):
            test_case = TestCase(
                title=f"{story.title} - {test_type.value.upper()} Test {i}",
                description=f"Verify: {criterion}",
                steps=[
                    "Setup test environment",
                    f"Execute test scenario for: {criterion}",
                    "Verify expected results",
                    "Cleanup test data"
                ],
                expected_result=criterion,
                test_type=test_type,
                priority=1 if story.priority <= 2 else 2,
                user_story_id=story.id,
                automated=True,
                automation_file=automation_file
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_e2e_test_cases_for_story(self, story: UserStory) -> List[TestCase]:
        """
        Generate E2E test cases specifically for user workflows.
        
        Args:
            story: User story with acceptance criteria
            
        Returns:
            List of E2E test cases
        """
        test_cases = []
        
        # Generate main happy path test
        main_test = TestCase(
            title=f"E2E: {story.title} - Happy Path",
            description=f"End-to-end test for: {story.description}",
            steps=[
                "Navigate to the application",
                "Login as test user" if story.persona else "Access the feature",
                f"Complete workflow: {story.title}",
                "Verify all acceptance criteria are met",
                "Verify data persistence",
                "Logout/cleanup"
            ],
            expected_result=f"User can successfully complete: {story.title}",
            test_type=TestType.E2E,
            priority=story.priority,
            user_story_id=story.id,
            automated=True,
            automation_file="tests/e2e/user_workflows.spec.ts"
        )
        test_cases.append(main_test)
        
        # Generate error handling test for high priority stories
        if story.priority <= 2:
            error_test = TestCase(
                title=f"E2E: {story.title} - Error Handling",
                description=f"Test error scenarios for: {story.description}",
                steps=[
                    "Navigate to the application",
                    "Access the feature",
                    "Trigger error conditions",
                    "Verify error messages are displayed",
                    "Verify application remains stable"
                ],
                expected_result="Application handles errors gracefully",
                test_type=TestType.E2E,
                priority=story.priority + 1,
                user_story_id=story.id,
                automated=True,
                automation_file="tests/e2e/error_scenarios.spec.ts"
            )
            test_cases.append(error_test)
        
        return test_cases
