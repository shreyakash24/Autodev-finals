"""
CrewAI Crew Configuration
Configures the multi-agent crew for the agentic code generation system
"""

from typing import List, Dict, Any, Optional
import os

from crewai import Crew, Process, Task as CrewTask
from langchain_openai import ChatOpenAI

from .agents import (
    ADOConnectorAgent, OrchestratorAgent,
    FrontendCodingAgent, BackendCodingAgent, DatabaseCodingAgent,
    TestingAgent, LegacyAnalyzerAgent, PromptRefinerAgent, MonitoringAgent
)
from .models import CanonicalSpec, PipelineState, Task, TaskStatus, GeneratedArtifact, AgentType


class CodeGenerationCrew:
    """
    Main crew that orchestrates all agents for code generation.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Code Generation Crew                                 │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Orchestrator                                   │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                              │                                           │
    │          ┌──────────────────┼──────────────────┐                        │
    │          │                  │                  │                        │
    │          ▼                  ▼                  ▼                        │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
    │  │  ADO Agent   │  │ Coding Agents│  │Testing Agent │                  │
    │  └──────────────┘  └──────────────┘  └──────────────┘                  │
    │                            │                  │                         │
    │                            ▼                  ▼                         │
    │                    ┌──────────────┐  ┌──────────────┐                  │
    │                    │Legacy Analyze│  │Prompt Refine │                  │
    │                    └──────────────┘  └──────────────┘                  │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Monitoring Agent                               │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, auto_mode: bool = True, ado_config: Optional[Dict[str, str]] = None):
        """
        Initialize the code generation crew.
        
        Args:
            auto_mode: Enable automatic integration of all agents (default: True)
            ado_config: Optional Azure DevOps configuration for auto-commit
                       {'org_url': '', 'pat': '', 'project': '', 'repo_name': ''}
        """
        # Initialize all agents
        self.ado_agent = ADOConnectorAgent()
        self.orchestrator = OrchestratorAgent()
        self.frontend_agent = FrontendCodingAgent()
        self.backend_agent = BackendCodingAgent()
        self.database_agent = DatabaseCodingAgent()
        self.testing_agent = TestingAgent()
        self.legacy_agent = LegacyAnalyzerAgent()
        self.prompt_agent = PromptRefinerAgent()
        self.monitoring_agent = MonitoringAgent()
        
        # Store current state
        self.current_spec: Optional[CanonicalSpec] = None
        self.current_pipeline: Optional[PipelineState] = None
        self.legacy_analysis: Optional[Dict[str, Any]] = None
        
        # Configuration
        self.auto_mode = auto_mode
        self.ado_config = ado_config or {}
        
        # Setup monitoring callbacks
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup monitoring callbacks."""
        def log_event(event):
            print(f"[{event.timestamp}] {event.event_type.value}: {event.data}")
        
        self.monitoring_agent.subscribe(log_event)
    
    def process_requirements(self, data: str, data_format: str = 'json',
                           tech_stack: Optional[Dict[str, str]] = None,
                           repo_path: Optional[str] = None) -> CanonicalSpec:
        """
        Process requirements from ADO data.
        
        Args:
            data: Raw requirements data (JSON or CSV)
            data_format: Format of input data
            tech_stack: Optional technology stack override
            repo_path: Optional path to legacy repository for analysis
            
        Returns:
            Canonical specification
        """
        # Step 1: Analyze legacy repository if provided and in auto mode
        if self.auto_mode and repo_path and os.path.exists(repo_path):
            print("[AutoMode] Analyzing legacy repository...")
            try:
                self.legacy_analysis = self.analyze_legacy(repo_path)
                print(f"[AutoMode] ✓ Legacy analysis complete: {self.legacy_analysis.get('tech_stack', {})}")
                
                # Use legacy tech stack if not explicitly provided
                if not tech_stack and self.legacy_analysis.get('tech_stack'):
                    tech_stack = self.legacy_analysis['tech_stack']
                    print(f"[AutoMode] Using detected tech stack from legacy repo")
            except Exception as e:
                print(f"[AutoMode] Legacy analysis failed (continuing anyway): {str(e)}")
        
        # Step 2: Process requirements with ADO agent
        self.current_spec = self.ado_agent.process(data, data_format, tech_stack)
        
        # Step 3: Refine user story descriptions if in auto mode
        if self.auto_mode:
            print("[AutoMode] Refining user story prompts...")
            for story in self.current_spec.user_stories:
                if story.description:
                    try:
                        # Refine the description
                        result = self.prompt_agent.refine_prompt(
                            story.description,
                            criteria=story.acceptance_criteria
                        )
                        if result.confidence_score > 0.7:  # Only use if confident
                            story.description = result.refined_prompt
                            print(f"[AutoMode] ✓ Refined story {story.id} (confidence: {result.confidence_score:.1%})")
                    except Exception as e:
                        print(f"[AutoMode] Prompt refinement failed for story {story.id} (continuing anyway): {str(e)}")
        
        return self.current_spec
    
    def build_pipeline(self, spec: Optional[CanonicalSpec] = None) -> PipelineState:
        """
        Build the execution pipeline.
        
        Args:
            spec: Canonical specification (uses current if not provided)
            
        Returns:
            Pipeline state with all tasks
        """
        spec = spec or self.current_spec
        if not spec:
            raise ValueError("No specification available. Process requirements first.")
        
        self.current_pipeline = self.orchestrator.build_pipeline(spec)
        self.monitoring_agent.set_pipeline(self.current_pipeline)
        
        return self.current_pipeline
    
    def execute_pipeline(self, parallel: bool = True, auto_commit: bool = None) -> PipelineState:
        """
        Execute the entire pipeline with dynamic enhancement from legacy analyzer,
        prompt refiner, and testing agent.
        
        Args:
            parallel: Whether to execute tasks in parallel where possible
            auto_commit: Whether to auto-commit to Azure DevOps (None = use self.auto_mode)
            
        Returns:
            Final pipeline state
        """
        if not self.current_pipeline:
            raise ValueError("No pipeline available. Build pipeline first.")
        
        # Determine if we should auto-commit
        should_auto_commit = auto_commit if auto_commit is not None else self.auto_mode
        
        # Execute the pipeline
        if parallel:
            result = self._execute_parallel()
        else:
            result = self._execute_sequential()
        
        # Generate comprehensive E2E tests for the complete application
        if self.auto_mode:
            print("[Pipeline] Generating comprehensive end-to-end tests...")
            self._generate_comprehensive_e2e_tests()
        
        # Add project structure files
        print("[Pipeline] Adding project structure files...")
        self._add_project_structure_files()
        
        # Auto-commit to Azure DevOps if enabled and configured
        commit_result = None
        if should_auto_commit and self.ado_config.get('org_url') and self.ado_config.get('pat'):
            print("[AutoMode] Pipeline execution complete. Auto-committing to Azure DevOps...")
            try:
                commit_result = self._auto_commit_to_ado()
                # Store commit result in pipeline state
                self.current_pipeline.commit_result = commit_result
            except Exception as e:
                print(f"[AutoMode] Auto-commit failed: {str(e)}")
                print("[AutoMode] You can manually commit using the UI or API")
                self.current_pipeline.commit_result = {"success": False, "error": str(e)}
        elif should_auto_commit:
            print("[AutoMode] Auto-commit enabled but Azure DevOps credentials not configured")
            print("[AutoMode] Set ado_config with org_url, pat, project, and repo_name to enable auto-commit")
        
        return result
    
    def _execute_parallel(self) -> PipelineState:
        """Execute pipeline with parallel task groups."""
        groups = self.orchestrator.get_parallel_groups()
        
        for group in groups:
            # Execute all tasks in group
            for task in group:
                self._execute_task(task)
        
        return self.current_pipeline
    
    def _execute_sequential(self) -> PipelineState:
        """Execute pipeline sequentially."""
        task_order = self.orchestrator.topological_sort()
        
        for task_id in task_order:
            task = next(
                (t for t in self.current_pipeline.tasks if t.id == task_id),
                None
            )
            if task:
                self._execute_task(task)
        
        return self.current_pipeline
    
    def _execute_task(self, task: Task) -> None:
        """
        Execute a single task with dynamic enhancement from legacy analyzer,
        prompt refiner, and testing agent.
        
        Args:
            task: Task to execute
        """
        # Update status
        self.orchestrator.update_task_status(task.id, TaskStatus.IN_PROGRESS)
        
        try:
            artifacts = []
            requirements = task.input_data.get('requirements', {})
            
            # Apply legacy analysis patterns if available
            if self.auto_mode and self.legacy_analysis:
                requirements = self._apply_legacy_patterns(requirements, task)
            
            # Auto-refine requirements prompt if enabled
            if self.auto_mode and requirements:
                requirements = self._refine_requirements_prompt(requirements, task)
            
            print(f"[Pipeline] Executing task {task.id} with agent {task.agent_type.value}")
            
            # Execute based on agent type
            if task.agent_type.value == 'frontend_coder':
                artifacts = self.frontend_agent.generate_component_scaffold(requirements)
                artifacts.extend(self.frontend_agent.generate_forms(requirements))
                artifacts.extend(self.frontend_agent.add_accessibility([]))
                
                # Generate frontend tests dynamically
                if self.auto_mode:
                    print(f"[Pipeline] Generating frontend tests for task {task.id}")
                    test_artifacts = self._generate_tests_for_artifacts(artifacts, 'frontend')
                    artifacts.extend(test_artifacts)
                
            elif task.agent_type.value == 'backend_coder':
                artifacts = self.backend_agent.generate_api_contracts(requirements)
                artifacts.extend(self.backend_agent.generate_services(requirements))
                artifacts.extend(self.backend_agent.generate_controllers(requirements))
                
                # Generate backend tests dynamically
                if self.auto_mode:
                    print(f"[Pipeline] Generating backend tests for task {task.id}")
                    test_artifacts = self._generate_tests_for_artifacts(artifacts, 'backend')
                    artifacts.extend(test_artifacts)
                
            elif task.agent_type.value == 'database_coder':
                artifacts = self.database_agent.generate_schema(requirements)
                artifacts.extend(self.database_agent.generate_orm_models(requirements))
                
                # Generate database tests dynamically
                if self.auto_mode:
                    print(f"[Pipeline] Generating database tests for task {task.id}")
                    test_artifacts = self._generate_tests_for_artifacts(artifacts, 'database')
                    artifacts.extend(test_artifacts)
                
            elif task.agent_type.value == 'testing':
                task_type = task.input_data.get('task_type', '')
                
                # Handle test plan generation
                if task_type == 'test_plan':
                    user_stories_data = task.input_data.get('user_stories', [])
                    project_name = task.input_data.get('project_name', 'Generated Code')
                    
                    # Convert user stories from dict to objects if needed
                    from .models import UserStory
                    user_stories = []
                    for story_data in user_stories_data:
                        if isinstance(story_data, dict):
                            story = UserStory(
                                id=story_data.get('id', ''),
                                title=story_data.get('title', ''),
                                description=story_data.get('description', ''),
                                acceptance_criteria=story_data.get('acceptance_criteria', []),
                                priority=story_data.get('priority', 3),
                                persona=story_data.get('persona')
                            )
                            user_stories.append(story)
                        else:
                            user_stories.append(story_data)
                    
                    # Generate test plan
                    test_plan = self.testing_agent.generate_test_plan(
                        user_stories=user_stories,
                        artifacts=self.orchestrator.pipeline_state.artifacts,
                        project_name=project_name
                    )
                    
                    # Add test plan to pipeline state
                    self.orchestrator.pipeline_state.test_plans.append(test_plan)
                    print(f"[Pipeline] Generated test plan with {len(test_plan.test_suites)} suites")
                    
                    # Don't add artifacts for test plan task
                    artifacts = []
                else:
                    # Handle test generation
                    test_type = task.input_data.get('test_type', 'unit')
                    if test_type == 'unit':
                        artifacts = self.testing_agent.generate_unit_tests(requirements, [])
                    elif test_type == 'integration':
                        artifacts = self.testing_agent.generate_integration_tests(requirements)
                    else:
                        artifacts = self.testing_agent.generate_e2e_tests(requirements)
            
            print(f"[Pipeline] Task {task.id} generated {len(artifacts)} artifacts")
            
            # Validate artifacts are not empty
            if not artifacts:
                print(f"[Pipeline] WARNING: Task {task.id} generated no artifacts")
            else:
                empty_count = 0
                for artifact in artifacts:
                    if not artifact.content or artifact.content.strip() == '':
                        print(f"[Pipeline] WARNING: Artifact {artifact.file_path} has empty content")
                        empty_count += 1
                    else:
                        print(f"[Pipeline] ✓ Artifact {artifact.file_path} ({len(artifact.content)} chars)")
                
                if empty_count > 0:
                    print(f"[Pipeline] WARNING: Task {task.id} generated {empty_count}/{len(artifacts)} empty artifacts")
            
            # Add artifacts to pipeline
            for artifact in artifacts:
                self.orchestrator.add_artifact(artifact)
            
            # Update task as completed
            self.orchestrator.update_task_status(
                task.id,
                TaskStatus.COMPLETED,
                output_data={'artifacts': [a.id for a in artifacts]}
            )
            
        except Exception as e:
            print(f"[Pipeline] ERROR: Task {task.id} failed: {str(e)}")
            self.orchestrator.update_task_status(
                task.id,
                TaskStatus.FAILED,
                error_message=str(e)
            )
    
    def _refine_requirements_prompt(self, requirements: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """
        Refine requirements prompt using the prompt refiner agent.
        
        Args:
            requirements: Original requirements
            task: Task being executed
            
        Returns:
            Refined requirements
        """
        try:
            # Build a prompt from requirements
            prompt_parts = []
            for key, value in requirements.items():
                if isinstance(value, str) and value:
                    prompt_parts.append(f"{key}: {value}")
            
            if not prompt_parts:
                return requirements
            
            prompt = "\n".join(prompt_parts)
            
            # Analyze for issues first
            issues = self.prompt_agent.analyze_prompt(prompt)
            
            # Only refine if there are issues
            if issues:
                print(f"[AutoMode] Found {len(issues)} issues in requirements for task {task.id}")
                result = self.prompt_agent.refine_prompt(prompt)
                
                if result.confidence_score > 0.6:  # Use if reasonably confident
                    # Update requirements with refined content
                    # This is a simple approach - in production you'd parse the refined prompt back
                    print(f"[AutoMode] ✓ Refined requirements (confidence: {result.confidence_score:.1%})")
                    # Add refined prompt as additional context
                    requirements['_refined_context'] = result.refined_prompt
                    
        except Exception as e:
            print(f"[AutoMode] Requirement refinement failed (continuing with original): {str(e)}")
        
        return requirements
    
    def _apply_legacy_patterns(self, requirements: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """
        Apply legacy code patterns to requirements for consistency with existing codebase.
        
        Args:
            requirements: Original requirements
            task: Task being executed
            
        Returns:
            Enhanced requirements with legacy patterns
        """
        try:
            if not self.legacy_analysis:
                return requirements
            
            print(f"[AutoMode] Applying legacy patterns to task {task.id}")
            
            # Extract relevant legacy patterns based on task type
            legacy_patterns = {}
            
            if task.agent_type.value == 'frontend_coder':
                # Apply frontend conventions
                if 'conventions' in self.legacy_analysis:
                    conventions = [c for c in self.legacy_analysis['conventions'] if 
                                 any(keyword in c.lower() for keyword in ['react', 'component', 'ui', 'css', 'style'])]
                    if conventions:
                        legacy_patterns['frontend_conventions'] = conventions
                
                # Apply tech stack patterns
                if 'tech_stack' in self.legacy_analysis:
                    tech = self.legacy_analysis['tech_stack']
                    if 'frontend' in tech or 'ui_framework' in tech:
                        legacy_patterns['legacy_frontend_stack'] = tech.get('frontend') or tech.get('ui_framework')
            
            elif task.agent_type.value == 'backend_coder':
                # Apply backend conventions
                if 'conventions' in self.legacy_analysis:
                    conventions = [c for c in self.legacy_analysis['conventions'] if 
                                 any(keyword in c.lower() for keyword in ['api', 'endpoint', 'service', 'controller', 'route'])]
                    if conventions:
                        legacy_patterns['backend_conventions'] = conventions
                
                if 'tech_stack' in self.legacy_analysis:
                    tech = self.legacy_analysis['tech_stack']
                    if 'backend' in tech:
                        legacy_patterns['legacy_backend_stack'] = tech.get('backend')
            
            elif task.agent_type.value == 'database_coder':
                # Apply database conventions
                if 'conventions' in self.legacy_analysis:
                    conventions = [c for c in self.legacy_analysis['conventions'] if 
                                 any(keyword in c.lower() for keyword in ['database', 'model', 'schema', 'migration', 'orm'])]
                    if conventions:
                        legacy_patterns['database_conventions'] = conventions
                
                if 'tech_stack' in self.legacy_analysis:
                    tech = self.legacy_analysis['tech_stack']
                    if 'database' in tech:
                        legacy_patterns['legacy_database'] = tech.get('database')
            
            if legacy_patterns:
                requirements['_legacy_patterns'] = legacy_patterns
                print(f"[AutoMode] ✓ Applied {len(legacy_patterns)} legacy pattern categories to task {task.id}")
            
        except Exception as e:
            print(f"[AutoMode] Legacy pattern application failed (continuing without): {str(e)}")
        
        return requirements
    
    def _generate_tests_for_artifacts(self, artifacts: List[GeneratedArtifact], context: str) -> List[GeneratedArtifact]:
        """
        Dynamically generate tests for code artifacts using the testing agent.
        
        Args:
            artifacts: List of code artifacts to test
            context: Context (frontend, backend, database) for test generation
            
        Returns:
            List of test artifacts
        """
        test_artifacts = []
        
        try:
            if not artifacts:
                return test_artifacts
            
            # Prepare requirements for test generation
            test_requirements = {
                'context': context,
                'artifact_count': len(artifacts),
                'artifact_types': list(set(a.artifact_type for a in artifacts)),
                'files_to_test': [a.file_path for a in artifacts[:5]]  # Limit to first 5 to avoid overwhelming
            }
            
            # Generate appropriate tests based on context
            if context == 'frontend':
                # Generate component tests
                test_artifacts = self.testing_agent.generate_unit_tests(test_requirements, artifacts)
                print(f"[AutoMode] ✓ Generated {len(test_artifacts)} frontend test files")
                
            elif context == 'backend':
                # Generate API and service tests
                unit_tests = self.testing_agent.generate_unit_tests(test_requirements, artifacts)
                integration_tests = self.testing_agent.generate_integration_tests(test_requirements)
                test_artifacts.extend(unit_tests)
                test_artifacts.extend(integration_tests)
                print(f"[AutoMode] ✓ Generated {len(test_artifacts)} backend test files (unit + integration)")
                
            elif context == 'database':
                # Generate model and schema tests
                test_artifacts = self.testing_agent.generate_unit_tests(test_requirements, artifacts)
                print(f"[AutoMode] ✓ Generated {len(test_artifacts)} database test files")
            
        except Exception as e:
            print(f"[AutoMode] Test generation failed for {context}: {str(e)}")
        
        return test_artifacts
    
    def _generate_comprehensive_e2e_tests(self) -> None:
        """
        Generate comprehensive end-to-end tests for the complete application.
        Tests all user stories and critical flows.
        """
        try:
            if not self.current_spec or not self.current_spec.user_stories:
                print("[AutoMode] No user stories available for E2E test generation")
                return
            
            # Build requirements from user stories for E2E testing
            e2e_requirements = {
                'user_stories': [
                    {
                        'id': story.id,
                        'title': story.title,
                        'description': story.description,
                        'acceptance_criteria': story.acceptance_criteria
                    }
                    for story in self.current_spec.user_stories
                ],
                'test_type': 'e2e',
                'tech_stack': self.current_spec.tech_stack
            }
            
            # Generate E2E tests
            e2e_artifacts = self.testing_agent.generate_e2e_tests(e2e_requirements)
            
            # Add E2E tests to pipeline
            for artifact in e2e_artifacts:
                self.orchestrator.add_artifact(artifact)
            
            print(f"[AutoMode] ✓ Generated {len(e2e_artifacts)} E2E test files covering {len(self.current_spec.user_stories)} user stories")
            
        except Exception as e:
            print(f"[AutoMode] Comprehensive E2E test generation failed: {str(e)}")
    
    def _add_project_structure_files(self) -> None:
        """Add project structure files (package.json, requirements.txt, README, etc.) to artifacts."""
        if not self.current_pipeline:
            return
        
        tech_stack = self.current_spec.tech_stack if self.current_spec else {}
        stories_summary = ""
        if self.current_spec and self.current_spec.user_stories:
            stories_summary = "\n".join([f"- {s.title}" for s in self.current_spec.user_stories[:10]])
        
        # Add package.json for frontend
        package_json = {
            "name": "generated-application",
            "version": "1.0.0",
            "description": "Auto-generated application from user stories",
            "main": "src/index.tsx",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "preview": "vite preview",
                "test": "vitest",
                "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.20.0",
                "axios": "^1.6.0"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "@typescript-eslint/eslint-plugin": "^6.0.0",
                "@typescript-eslint/parser": "^6.0.0",
                "@vitejs/plugin-react": "^4.2.0",
                "eslint": "^8.55.0",
                "eslint-plugin-react-hooks": "^4.6.0",
                "eslint-plugin-react-refresh": "^0.4.5",
                "typescript": "^5.2.2",
                "vite": "^5.0.0",
                "vitest": "^1.0.0"
            }
        }
        
        import json
        package_artifact = GeneratedArtifact(
            file_path="package.json",
            content=json.dumps(package_json, indent=2),
            artifact_type="config",
            language="json",
            agent_type=AgentType.ORCHESTRATOR,
            documentation="NPM package configuration for the frontend application"
        )
        self.orchestrator.add_artifact(package_artifact)
        print(f"[Pipeline] ✓ Added package.json")
        
        # Add requirements.txt for backend
        requirements_txt = """# Backend Dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
"""
        
        requirements_artifact = GeneratedArtifact(
            file_path="requirements.txt",
            content=requirements_txt,
            artifact_type="config",
            language="text",
            agent_type=AgentType.ORCHESTRATOR,
            documentation="Python dependencies for the backend application"
        )
        self.orchestrator.add_artifact(requirements_artifact)
        print(f"[Pipeline] ✓ Added requirements.txt")
        
        # Add README.md
        readme_content = f"""# Generated Application

This application was automatically generated from user stories using the Agentic Code Generation System.

## Features

{stories_summary if stories_summary else "- Auto-generated features based on user requirements"}

## Tech Stack

- **Frontend**: {tech_stack.get('frontend', 'React')} with TypeScript
- **Backend**: {tech_stack.get('backend', 'FastAPI')} (Python)
- **Database**: {tech_stack.get('database', 'PostgreSQL')}
- **Testing**: {tech_stack.get('testing', 'pytest')} + Vitest

## Getting Started

### Frontend

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload
```

## Project Structure

```
.
├── src/                    # Frontend source code
│   ├── components/         # React components
│   ├── pages/             # Page components
│   ├── services/          # API services
│   └── utils/             # Utility functions
├── backend/               # Backend source code
│   ├── api/              # API routes
│   ├── models/           # Database models
│   ├── services/         # Business logic
│   └── tests/            # Backend tests
├── tests/                # Frontend tests
├── package.json          # Frontend dependencies
├── requirements.txt      # Backend dependencies
└── README.md            # This file
```

## Testing

### Frontend Tests
```bash
npm test
```

### Backend Tests
```bash
pytest
```

## Documentation

For more information about the Agentic Code Generation System, see the main repository documentation.

## License

MIT License
"""
        
        readme_artifact = GeneratedArtifact(
            file_path="README.md",
            content=readme_content,
            artifact_type="documentation",
            language="markdown",
            agent_type=AgentType.ORCHESTRATOR,
            documentation="Project documentation and setup instructions"
        )
        self.orchestrator.add_artifact(readme_artifact)
        print(f"[Pipeline] ✓ Added README.md")
        
        # Add .gitignore
        gitignore_content = """# Dependencies
node_modules/
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
pip-log.txt
pip-delete-this-directory.txt

# Build outputs
dist/
build/
*.egg-info/
.eggs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
.env.*.local

# Testing
coverage/
.coverage
htmlcov/
.pytest_cache/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
"""
        
        gitignore_artifact = GeneratedArtifact(
            file_path=".gitignore",
            content=gitignore_content,
            artifact_type="config",
            language="text",
            agent_type=AgentType.ORCHESTRATOR,
            documentation="Git ignore configuration"
        )
        self.orchestrator.add_artifact(gitignore_artifact)
        print(f"[Pipeline] ✓ Added .gitignore")
        
        # Add vite.config.ts for Vite frontend setup
        vite_config = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
"""
        
        vite_artifact = GeneratedArtifact(
            file_path="vite.config.ts",
            content=vite_config,
            artifact_type="config",
            language="typescript",
            agent_type=AgentType.ORCHESTRATOR,
            documentation="Vite configuration for frontend build"
        )
        self.orchestrator.add_artifact(vite_artifact)
        print(f"[Pipeline] ✓ Added vite.config.ts")
        
        # Add tsconfig.json
        tsconfig = """{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
"""
        
        tsconfig_artifact = GeneratedArtifact(
            file_path="tsconfig.json",
            content=tsconfig,
            artifact_type="config",
            language="json",
            agent_type=AgentType.ORCHESTRATOR,
            documentation="TypeScript configuration"
        )
        self.orchestrator.add_artifact(tsconfig_artifact)
        print(f"[Pipeline] ✓ Added tsconfig.json")
        
        # Add backend main.py entry point
        main_py = """\"\"\"
Backend API Entry Point
\"\"\"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Generated Application API",
    description="Auto-generated API from user stories",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the generated API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Import and include routers from generated API modules
# from .api import router
# app.include_router(router, prefix="/api")
"""
        
        main_py_artifact = GeneratedArtifact(
            file_path="backend/main.py",
            content=main_py,
            artifact_type="backend",
            language="python",
            agent_type=AgentType.ORCHESTRATOR,
            documentation="Backend API entry point"
        )
        self.orchestrator.add_artifact(main_py_artifact)
        print(f"[Pipeline] ✓ Added backend/main.py")
        
        print(f"[Pipeline] Added {7} project structure files to artifacts")
    
    def _auto_commit_to_ado(self) -> Dict[str, Any]:
        """
        Automatically publish all generated artifacts to Azure DevOps.
        This includes:
        - Azure Repos: Code artifacts
        - Azure Test Plans: Test plans
        - Azure Boards: Work items
        
        Returns:
            Publication result
        """
        if not self.current_pipeline or not self.current_pipeline.artifacts:
            print("[AutoMode] No artifacts to publish")
            return {"success": False, "error": "No artifacts available"}
        
        # Filter out empty artifacts
        valid_artifacts = [
            a for a in self.current_pipeline.artifacts 
            if a.content and a.content.strip()
        ]
        
        if not valid_artifacts:
            print("[AutoMode] All artifacts are empty, skipping publication")
            return {"success": False, "error": "All artifacts are empty"}
        
        print(f"[AutoMode] Publishing {len(valid_artifacts)} artifacts to Azure DevOps...")
        
        # Update pipeline with valid artifacts
        self.current_pipeline.artifacts = valid_artifacts
        
        # Create ADO agent with credentials
        temp_agent = ADOConnectorAgent(
            ado_url=self.ado_config.get('org_url'),
            pat=self.ado_config.get('pat')
        )
        temp_agent.project = self.ado_config.get('project')
        
        # Publish to all Azure DevOps sections
        result = temp_agent.publish_to_azure_devops(
            pipeline_state=self.current_pipeline,
            repo_name=self.ado_config.get('repo_name', 'generated-code'),
            branch=self.ado_config.get('branch', 'refs/heads/generated-code')
        )
        
        if result.get('success'):
            print(f"[AutoMode] ✓ Successfully published to Azure DevOps")
            if result.get('repos'):
                print(f"[AutoMode]   - Repos: {result['repos'].get('files_committed', 0)} files committed")
            if result.get('test_plans'):
                print(f"[AutoMode]   - Test Plans: {len(result['test_plans'])} plans created")
            if result.get('boards'):
                print(f"[AutoMode]   - Boards: {result['boards'].get('work_items_created', 0)} work items created")
        else:
            print(f"[AutoMode] ✗ Publication had errors")
            if result.get('errors'):
                for error in result['errors']:
                    print(f"[AutoMode]     - {error}")
        
        return result
    
    def run_tests(self, test_type: str = 'all') -> Dict[str, Any]:
        """
        Run tests and return results.
        
        Args:
            test_type: Type of tests to run ('unit', 'integration', 'e2e', 'all')
            
        Returns:
            Test results summary
        """
        from .models import TestType
        
        results = {}
        
        if test_type in ('unit', 'all'):
            report = self.testing_agent.run_tests('tests/', TestType.UNIT)
            self.orchestrator.add_test_report(report)
            results['unit'] = report.to_dict()
        
        if test_type in ('integration', 'all'):
            report = self.testing_agent.run_tests('tests/', TestType.INTEGRATION)
            self.orchestrator.add_test_report(report)
            results['integration'] = report.to_dict()
        
        if test_type in ('e2e', 'all'):
            report = self.testing_agent.run_tests('tests/e2e/', TestType.E2E)
            self.orchestrator.add_test_report(report)
            results['e2e'] = report.to_dict()
        
        return results
    
    def analyze_legacy(self, repo_path: str) -> Dict[str, Any]:
        """
        Analyze a legacy repository and store results for use in code generation.
        
        Args:
            repo_path: Path to legacy repository
            
        Returns:
            Analysis results
        """
        analysis = self.legacy_agent.analyze_repository(repo_path)
        analysis_dict = analysis.to_dict()
        
        # Store analysis for use in code generation
        self.legacy_analysis = analysis_dict
        print(f"[AutoMode] ✓ Legacy analysis stored: {len(analysis_dict.get('conventions', []))} conventions, {len(analysis_dict.get('dependencies', []))} dependencies")
        
        return analysis_dict
    
    def refine_prompt(self, prompt: str, 
                     criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Refine a prompt.
        
        Args:
            prompt: Prompt to refine
            criteria: Optional acceptance criteria
            
        Returns:
            Refinement results
        """
        result = self.prompt_agent.refine_prompt(prompt, criteria)
        return {
            'original': result.original_prompt,
            'refined': result.refined_prompt,
            'improvements': result.improvements,
            'confidence': result.confidence_score
        }
    
    def check_quality_gate(self) -> Dict[str, Any]:
        """
        Check if quality gates are met.
        
        Returns:
            Quality gate status
        """
        return self.orchestrator.check_quality_gate()
    
    def create_checkpoint(self, name: str) -> str:
        """
        Create a checkpoint for rollback.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Checkpoint ID
        """
        if not self.current_pipeline:
            raise ValueError("No pipeline to checkpoint")
        
        checkpoint = self.monitoring_agent.create_checkpoint(name, self.current_pipeline)
        return checkpoint.id
    
    def rollback(self, checkpoint_id: str) -> bool:
        """
        Rollback to a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint
            
        Returns:
            True if successful
        """
        result = self.monitoring_agent.rollback_to_checkpoint(checkpoint_id)
        return result is not None
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for the UI dashboard.
        
        Returns:
            Dashboard data
        """
        return self.monitoring_agent.get_dashboard_data()
    
    def get_artifacts(self) -> List[Dict[str, Any]]:
        """
        Get all generated artifacts.
        
        Returns:
            List of artifact dictionaries
        """
        if not self.current_pipeline:
            return []
        return [a.to_dict() for a in self.current_pipeline.artifacts]
    
    def configure_ado(self, org_url: str, pat: str, project: str, repo_name: str,
                      branch: str = 'refs/heads/generated-code',
                      commit_message: Optional[str] = None):
        """
        Configure Azure DevOps settings for auto-commit.
        
        Args:
            org_url: Azure DevOps organization URL
            pat: Personal Access Token
            project: Project name
            repo_name: Repository name
            branch: Branch name (default: refs/heads/generated-code)
            commit_message: Optional custom commit message
        """
        self.ado_config = {
            'org_url': org_url,
            'pat': pat,
            'project': project,
            'repo_name': repo_name,
            'branch': branch
        }
        if commit_message:
            self.ado_config['commit_message'] = commit_message
        print(f"[Config] Azure DevOps configured for auto-commit to {repo_name}")
    
    def set_auto_mode(self, enabled: bool):
        """
        Enable or disable automatic mode.
        
        Args:
            enabled: True to enable auto mode, False to disable
        """
        self.auto_mode = enabled
        mode_str = "enabled" if enabled else "disabled"
        print(f"[Config] Automatic mode {mode_str}")



def create_crew(auto_mode: bool = True, ado_config: Optional[Dict[str, str]] = None) -> CodeGenerationCrew:
    """
    Factory function to create a code generation crew.
    
    Args:
        auto_mode: Enable automatic integration of all agents (default: True)
        ado_config: Optional Azure DevOps configuration for auto-commit
    
    Returns:
        Configured CodeGenerationCrew instance
    """
    return CodeGenerationCrew(auto_mode=auto_mode, ado_config=ado_config)
