import os
import re
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import Counter

from crewai import Agent

from ..utils.llm_config import get_llm

from ..models import LegacyAnalysis, AgentType


class LegacyAnalyzerAgent:
    TECH_PATTERNS = {
        # Python
        'requirements.txt': {'language': 'Python', 'package_manager': 'pip'},
        'setup.py': {'language': 'Python', 'package_manager': 'pip'},
        'pyproject.toml': {'language': 'Python', 'package_manager': 'poetry'},
        'Pipfile': {'language': 'Python', 'package_manager': 'pipenv'},
        
        # JavaScript/TypeScript
        'package.json': {'language': 'JavaScript', 'package_manager': 'npm'},
        'yarn.lock': {'language': 'JavaScript', 'package_manager': 'yarn'},
        'tsconfig.json': {'language': 'TypeScript'},
        
        # Java
        'pom.xml': {'language': 'Java', 'package_manager': 'maven'},
        'build.gradle': {'language': 'Java', 'package_manager': 'gradle'},
        
        # Go
        'go.mod': {'language': 'Go', 'package_manager': 'go modules'},
        
        # Ruby
        'Gemfile': {'language': 'Ruby', 'package_manager': 'bundler'},
        
        # .NET
        '*.csproj': {'language': 'C#', 'package_manager': 'nuget'},
        
        # Rust
        'Cargo.toml': {'language': 'Rust', 'package_manager': 'cargo'},
    }
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        # Python frameworks
        'django': 'Django',
        'flask': 'Flask',
        'fastapi': 'FastAPI',
        'pyramid': 'Pyramid',
        'tornado': 'Tornado',
        
        # JavaScript frameworks
        'react': 'React',
        'vue': 'Vue.js',
        'angular': 'Angular',
        'express': 'Express.js',
        'next': 'Next.js',
        'nest': 'NestJS',
        
        # Java frameworks
        'spring': 'Spring',
        'hibernate': 'Hibernate',
        
        # Database
        'sqlalchemy': 'SQLAlchemy',
        'mongoose': 'Mongoose',
        'sequelize': 'Sequelize',
        'prisma': 'Prisma',
    }
    
    # Architecture patterns
    ARCHITECTURE_PATTERNS = {
        'mvc': ['controllers', 'views', 'models'],
        'layered': ['services', 'repositories', 'controllers'],
        'microservices': ['docker-compose', 'kubernetes', 'k8s'],
        'hexagonal': ['adapters', 'ports', 'domain'],
        'clean': ['usecases', 'entities', 'interfaces'],
        'ddd': ['domain', 'application', 'infrastructure'],
    }
    
    def __init__(self):
        self.agent_type = AgentType.LEGACY_ANALYZER
        self.llm = get_llm(temperature=0.1)
        self.crew_agent = Agent(
            role="Legacy System Analyst",
            goal="Analyze legacy codebases and propose integration strategies",
            backstory="""You are an expert at analyzing legacy systems and
            understanding their architecture. You identify patterns, conventions,
            and potential integration points for new features.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        ) if self.llm else None
    
    def analyze_repository(self, repo_path: str) -> LegacyAnalysis:
        if not os.path.exists(repo_path):
            return LegacyAnalysis(
                repo_path=repo_path,
                compatibility_issues=["Repository path does not exist"]
            )
        
        files = self._scan_files(repo_path)
        tech_stack = self._detect_tech_stack(repo_path, files)
        
        # Analyze dependencies
        dependencies = self._analyze_dependencies(repo_path, tech_stack)
        
        # Detect architecture
        architecture = self._detect_architecture(files)
        
        # Detect conventions
        conventions = self._detect_conventions(repo_path, files, tech_stack)
        
        # Generate integration strategy
        integration_strategy = self._propose_integration_strategy(
            tech_stack, architecture, dependencies
        )
        
        # Check compatibility
        compatibility_issues = self._check_compatibility(tech_stack, dependencies)
        
        return LegacyAnalysis(
            repo_path=repo_path,
            tech_stack=tech_stack,
            architecture=architecture,
            dependencies=dependencies,
            conventions=conventions,
            integration_strategy=integration_strategy,
            compatibility_issues=compatibility_issues
        )
    
    def _scan_files(self, repo_path: str) -> List[str]:
        files = []
        exclude_dirs = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 
                       'dist', 'build', '.idea', '.vscode'}
        
        for root, dirs, filenames in os.walk(repo_path):
            # Exclude certain directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for filename in filenames:
                rel_path = os.path.relpath(
                    os.path.join(root, filename), 
                    repo_path
                )
                files.append(rel_path)
        
        return files
    
    def _detect_tech_stack(self, repo_path: str, files: List[str]) -> Dict[str, str]:
        tech_stack = {}
        
        # Check for known config files
        for pattern, tech in self.TECH_PATTERNS.items():
            if pattern.startswith('*'):
                # Wildcard pattern
                ext = pattern[1:]
                if any(f.endswith(ext) for f in files):
                    tech_stack.update(tech)
            elif pattern in files:
                tech_stack.update(tech)
        
        # Detect frameworks from dependencies
        if 'package.json' in files:
            frameworks = self._detect_npm_frameworks(
                os.path.join(repo_path, 'package.json')
            )
            tech_stack.update(frameworks)
        
        if 'requirements.txt' in files:
            frameworks = self._detect_pip_frameworks(
                os.path.join(repo_path, 'requirements.txt')
            )
            tech_stack.update(frameworks)
        
        # Detect database from files
        db = self._detect_database(files, repo_path)
        if db:
            tech_stack['database'] = db
        
        # Detect frontend framework
        if any(f.endswith('.tsx') or f.endswith('.jsx') for f in files):
            tech_stack['frontend'] = tech_stack.get('frontend', 'React')
        elif any(f.endswith('.vue') for f in files):
            tech_stack['frontend'] = 'Vue.js'
        
        return tech_stack
    
    def _detect_npm_frameworks(self, package_json_path: str) -> Dict[str, str]:
        frameworks = {}
        
        try:
            import json
            with open(package_json_path, 'r') as f:
                pkg = json.load(f)
            
            all_deps = {
                **pkg.get('dependencies', {}),
                **pkg.get('devDependencies', {})
            }
            
            for dep_name in all_deps.keys():
                dep_lower = dep_name.lower()
                for pattern, framework in self.FRAMEWORK_PATTERNS.items():
                    if pattern in dep_lower:
                        if 'frontend' not in frameworks:
                            frameworks['frontend'] = framework
                        elif 'backend' not in frameworks:
                            frameworks['backend'] = framework
                        break
                        
        except Exception:
            pass
        
        return frameworks
    
    def _detect_pip_frameworks(self, requirements_path: str) -> Dict[str, str]:
        frameworks = {}
        
        try:
            with open(requirements_path, 'r') as f:
                requirements = f.read().lower()
            
            for pattern, framework in self.FRAMEWORK_PATTERNS.items():
                if pattern in requirements:
                    if 'backend' not in frameworks:
                        frameworks['backend'] = framework
                    break
                        
        except Exception:
            pass
        
        return frameworks
    
    def _detect_database(self, files: List[str], repo_path: str) -> Optional[str]:
        # Check for database configuration files
        db_indicators = {
            'postgresql': ['postgres', 'psycopg2', 'pg_'],
            'mysql': ['mysql', 'pymysql'],
            'mongodb': ['mongo', 'mongoose'],
            'sqlite': ['sqlite'],
            'redis': ['redis'],
        }
        
        # Check configuration files
        config_files = [f for f in files if 'config' in f.lower() or f.endswith('.env')]
        
        for config_file in config_files:
            try:
                full_path = os.path.join(repo_path, config_file)
                with open(full_path, 'r') as f:
                    content = f.read().lower()
                
                for db, indicators in db_indicators.items():
                    if any(ind in content for ind in indicators):
                        return db
            except Exception:
                continue
        
        return None
    
    def _analyze_dependencies(self, repo_path: str, 
                             tech_stack: Dict[str, str]) -> List[Dict[str, str]]:
        dependencies = []
        
        # Parse requirements.txt
        req_path = os.path.join(repo_path, 'requirements.txt')
        if os.path.exists(req_path):
            dependencies.extend(self._parse_requirements(req_path))
        
        # Parse package.json
        pkg_path = os.path.join(repo_path, 'package.json')
        if os.path.exists(pkg_path):
            dependencies.extend(self._parse_package_json(pkg_path))
        
        return dependencies
    
    def _parse_requirements(self, req_path: str) -> List[Dict[str, str]]:
        dependencies = []
        
        try:
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package==version or package>=version
                        match = re.match(r'^([a-zA-Z0-9_-]+)([<>=!]+)?(.+)?$', line)
                        if match:
                            dependencies.append({
                                'name': match.group(1),
                                'version': match.group(3) or 'any',
                                'type': 'python'
                            })
        except Exception:
            pass
        
        return dependencies
    
    def _parse_package_json(self, pkg_path: str) -> List[Dict[str, str]]:
        dependencies = []
        
        try:
            import json
            with open(pkg_path, 'r') as f:
                pkg = json.load(f)
            
            for name, version in pkg.get('dependencies', {}).items():
                dependencies.append({
                    'name': name,
                    'version': version,
                    'type': 'npm'
                })
            
            for name, version in pkg.get('devDependencies', {}).items():
                dependencies.append({
                    'name': name,
                    'version': version,
                    'type': 'npm-dev'
                })
        except Exception:
            pass
        
        return dependencies
    
    def _detect_architecture(self, files: List[str]) -> str:
        # Convert file paths to directory names
        dirs = set()
        for f in files:
            parts = Path(f).parts
            dirs.update(p.lower() for p in parts)
        
        # Check each architecture pattern
        scores = {}
        for arch, indicators in self.ARCHITECTURE_PATTERNS.items():
            score = sum(1 for ind in indicators if ind in dirs)
            if score > 0:
                scores[arch] = score
        
        if scores:
            # Return architecture with highest score
            return max(scores, key=scores.get)
        
        # Default based on common patterns
        if 'src' in dirs and 'tests' in dirs:
            return 'standard'
        
        return 'unknown'
    
    def _detect_conventions(self, repo_path: str, files: List[str],
                           tech_stack: Dict[str, str]) -> List[str]:
        conventions = []
        
        # Check for linter configurations
        linter_configs = {
            '.eslintrc': 'ESLint for JavaScript',
            '.eslintrc.json': 'ESLint for JavaScript',
            '.pylintrc': 'Pylint for Python',
            'pyproject.toml': 'Black/Ruff for Python',
            '.prettierrc': 'Prettier for formatting',
            '.editorconfig': 'EditorConfig',
            'tslint.json': 'TSLint for TypeScript',
        }
        
        for config, convention in linter_configs.items():
            if config in files:
                conventions.append(convention)
        
        # Check for test conventions
        test_patterns = {
            'test_': 'Pytest naming convention',
            '_test.py': 'Pytest naming convention',
            '.spec.ts': 'Jest/Mocha naming convention',
            '.test.ts': 'Jest naming convention',
        }
        
        for pattern, convention in test_patterns.items():
            if any(pattern in f for f in files):
                conventions.append(convention)
                break
        
        # Check for typing
        if any(f.endswith('.pyi') for f in files) or 'py.typed' in files:
            conventions.append('Python type hints')
        
        if 'tsconfig.json' in files:
            conventions.append('TypeScript strict mode')
        
        # Check for documentation conventions
        if 'docs' in [Path(f).parts[0] for f in files if '/' in f]:
            conventions.append('Documentation in docs/ folder')
        
        if any(f.endswith('.md') for f in files):
            conventions.append('Markdown documentation')
        
        # Check naming conventions from sample files
        naming_convention = self._detect_naming_convention(repo_path, files)
        if naming_convention:
            conventions.append(naming_convention)
        
        return conventions
    
    def _detect_naming_convention(self, repo_path: str, files: List[str]) -> Optional[str]:
        python_files = [f for f in files if f.endswith('.py')]
        js_files = [f for f in files if f.endswith(('.js', '.ts', '.jsx', '.tsx'))]
        
        if python_files:
            # Check if using snake_case (standard for Python)
            if all(re.match(r'^[a-z0-9_/\\]+\.py$', f) for f in python_files[:10]):
                return 'snake_case for Python files'
        
        if js_files:
            # Check for PascalCase components
            if any(re.match(r'.*[A-Z][a-z]+[A-Z].*\.(jsx|tsx)$', f) for f in js_files):
                return 'PascalCase for React components'
            # Check for camelCase
            if any(re.match(r'.*[a-z]+[A-Z][a-z]+.*\.(js|ts)$', f) for f in js_files):
                return 'camelCase for JavaScript files'
        
        return None
    
    def _propose_integration_strategy(self, tech_stack: Dict[str, str],
                                      architecture: str,
                                      dependencies: List[Dict[str, str]]) -> str:
        strategies = []
        
        # Based on architecture
        if architecture == 'microservices':
            strategies.append("Create new service with API gateway integration")
            strategies.append("Use message queue for async communication")
        elif architecture == 'mvc':
            strategies.append("Add new controllers following existing pattern")
            strategies.append("Extend models for new features")
        elif architecture == 'layered':
            strategies.append("Add new service layer for business logic")
            strategies.append("Create repository interfaces for data access")
        elif architecture in ('hexagonal', 'clean', 'ddd'):
            strategies.append("Define ports/interfaces for new features")
            strategies.append("Create adapters to connect with existing infrastructure")
        else:
            strategies.append("Create adapter layer for new functionality")
            strategies.append("Use facade pattern to simplify integration")
        
        # Based on tech stack
        backend = tech_stack.get('backend', '')
        if backend:
            strategies.append(f"Follow {backend} conventions for new endpoints")
        
        frontend = tech_stack.get('frontend', '')
        if frontend:
            strategies.append(f"Create {frontend} components matching existing style")
        
        # Build strategy string
        strategy = "Integration Strategy:\n"
        for i, s in enumerate(strategies, 1):
            strategy += f"{i}. {s}\n"
        
        return strategy
    
    def _check_compatibility(self, tech_stack: Dict[str, str],
                            dependencies: List[Dict[str, str]]) -> List[str]:
        issues = []
        
        # Check for outdated Python version indicators
        python_deps = [d for d in dependencies if d['type'] == 'python']
        for dep in python_deps:
            if dep['name'] == 'python' and dep['version']:
                try:
                    version = dep['version'].replace('>=', '').replace('==', '').split('.')[0:2]
                    if len(version) >= 2 and int(version[0]) == 2:
                        issues.append("Legacy Python 2.x detected - migration to Python 3 recommended")
                except Exception:
                    pass
        
        # Check for deprecated packages
        deprecated_packages = {
            'flask-script': 'Flask-Script is deprecated, use Flask CLI',
            'nose': 'Nose is deprecated, use pytest',
            'optparse': 'optparse is deprecated, use argparse',
        }
        
        for dep in dependencies:
            if dep['name'].lower() in deprecated_packages:
                issues.append(deprecated_packages[dep['name'].lower()])
        
        # Check for version conflicts potential
        npm_deps = [d for d in dependencies if d['type'] == 'npm']
        if any('^' not in d['version'] and '~' not in d['version'] 
               for d in npm_deps if d['version'] and d['version'] != 'any'):
            issues.append("Exact version pinning in package.json may cause conflicts")
        
        return issues
    
    def generate_migration_plan(self, analysis: LegacyAnalysis,
                               target_stack: Dict[str, str]) -> str:
        plan = ["Migration Plan", "=" * 50, ""]
        
        # Compare stacks
        current = analysis.tech_stack
        
        # Language migration
        if current.get('language') != target_stack.get('language'):
            plan.append(f"1. Language Migration: {current.get('language', 'Unknown')} -> {target_stack.get('language', 'Unknown')}")
            plan.append("   - Identify core business logic for rewrite")
            plan.append("   - Create adapter interfaces for gradual migration")
            plan.append("   - Set up parallel running for validation")
            plan.append("")
        
        # Framework migration
        if current.get('backend') != target_stack.get('backend'):
            plan.append(f"2. Backend Framework: {current.get('backend', 'Unknown')} -> {target_stack.get('backend', 'Unknown')}")
            plan.append("   - Map existing routes to new framework")
            plan.append("   - Migrate middleware and authentication")
            plan.append("   - Update request/response handling")
            plan.append("")
        
        if current.get('frontend') != target_stack.get('frontend'):
            plan.append(f"3. Frontend Framework: {current.get('frontend', 'Unknown')} -> {target_stack.get('frontend', 'Unknown')}")
            plan.append("   - Create component mapping document")
            plan.append("   - Migrate state management")
            plan.append("   - Update build configuration")
            plan.append("")
        
        # Database migration
        if current.get('database') != target_stack.get('database'):
            plan.append(f"4. Database: {current.get('database', 'Unknown')} -> {target_stack.get('database', 'Unknown')}")
            plan.append("   - Create schema migration scripts")
            plan.append("   - Plan data migration strategy")
            plan.append("   - Set up replication for zero-downtime migration")
            plan.append("")
        
        # Timeline estimation
        plan.append("Estimated Timeline:")
        plan.append("-" * 30)
        plan.append("- Phase 1 (Assessment): 1-2 weeks")
        plan.append("- Phase 2 (Parallel Development): 4-8 weeks")
        plan.append("- Phase 3 (Migration): 2-4 weeks")
        plan.append("- Phase 4 (Validation): 1-2 weeks")
        
        return "\n".join(plan)
