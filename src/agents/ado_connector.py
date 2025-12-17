"""
ADO Connector & Parser Agent
Connects to Azure DevOps via REST API or consumes exported JSON/CSV.
Extracts user stories, acceptance criteria, personas, and non-functional hints.
Normalizes requirements into a canonical spec for downstream agents.
"""

import json
import csv
import re
import base64
from typing import List, Dict, Any, Optional, Union, Tuple
from io import StringIO
import os

try:
    import requests
except ImportError:
    requests = None

from crewai import Agent, Task as CrewTask

from ..utils.llm_config import get_llm

from ..models import PipelineState, TestPlan, UserStory, CanonicalSpec, AgentType

# Constants for Azure DevOps validation
INVALID_PATH_CHARS = ['<', '>', ':', '"', '|', '?', '*']  # Azure DevOps restricted characters
MAX_FILE_SIZE_MB = 100  # Azure DevOps file size limit
MAX_COMMIT_MESSAGE_LENGTH = 4000  # Azure DevOps commit message limit
MAX_WORK_ITEMS_PER_COMPONENT = 5  # Limit feature work items to prevent overwhelming boards


class ADOConnectorAgent:
    """
    Agent responsible for connecting to ADO and parsing requirements.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    ADO Connector Agent                       │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ REST Client │  │ CSV Parser  │  │ JSON Parser         │  │
    │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
    │         │                │                     │             │
    │         └────────────────┴─────────────────────┘             │
    │                          │                                   │
    │                   ┌──────▼──────┐                           │
    │                   │ Normalizer  │                           │
    │                   └──────┬──────┘                           │
    │                          │                                   │
    │                   ┌──────▼──────┐                           │
    │                   │CanonicalSpec│                           │
    │                   └─────────────┘                           │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, ado_url: Optional[str] = None, pat: Optional[str] = None, project: Optional[str] = None):
        """
        Initialize the ADO Connector Agent.
        
        Args:
            ado_url: Azure DevOps organization URL
            pat: Personal Access Token for authentication
        """
        self.ado_url = ado_url or os.getenv('ADO_ORG_URL', '')
        self.pat = pat or os.getenv('ADO_PAT', '')
        self.project =project or os.getenv('ADO_PROJECT', '')
        
        # Initialize CrewAI agent
        self.llm = get_llm(temperature=0.1)
        
        self.crew_agent = Agent(
            role="Requirements Analyst",
            goal="Extract and normalize user stories from ADO data",
            backstory="""You are an expert at analyzing software requirements.
            You extract user stories, acceptance criteria, and identify
            non-functional requirements from various data sources.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        ) if self.llm else None
    
    def parse_json(self, json_data: str) -> List[UserStory]:
        """
        Parse user stories from JSON data.
        
        Args:
            json_data: JSON string containing work items
            
        Returns:
            List of UserStory objects
        """
        stories = []
        parse_errors = []
        
        try:
            data = json.loads(json_data)
            
            # Handle different JSON structures
            items = data if isinstance(data, list) else data.get('workItems', data.get('value', []))
            
            # Validate items structure
            if not items:
                raise ValueError(
                    "No work items found in JSON. Expected an array of work items or an object with 'workItems' or 'value' property.\n\n"
                    "Example format:\n"
                    "[\n"
                    "  {\n"
                    "    \"id\": \"1\",\n"
                    "    \"title\": \"User Story Title\",\n"
                    "    \"description\": \"As a user, I want to...\",\n"
                    "    \"acceptance_criteria\": [\"Criteria 1\", \"Criteria 2\"]\n"
                    "  }\n"
                    "]"
                )
            
            if not isinstance(items, list):
                raise ValueError(
                    "Work items must be an array.\n\n"
                    "Expected format: [{\"id\": \"1\", \"title\": \"Story\", ...}]"
                )
            
            for idx, item in enumerate(items):
                try:
                    story = self._parse_work_item(item)
                    if story:
                        stories.append(story)
                    else:
                        parse_errors.append(f"Item {idx + 1}: Could not parse work item (missing required fields)")
                except Exception as e:
                    parse_errors.append(f"Item {idx + 1}: {str(e)}")
            
            # Be lenient - if we parsed at least one story, consider it a success
            if not stories:
                error_details = "\n".join(parse_errors[:10]) if parse_errors else "Unknown parsing errors"
                if len(parse_errors) > 10:
                    error_details += f"\n... and {len(parse_errors) - 10} more errors"
                    
                raise ValueError(
                    "Failed to parse any work items. Please check that your JSON has the required fields.\n\n"
                    "Parsing errors:\n"
                    f"{error_details}\n\n"
                    "Each work item should have:\n"
                    "- id: Unique identifier (required)\n"
                    "- title: Story title (required)\n"
                    "- description: Story description (optional)\n"
                    "- acceptance_criteria: Array of criteria (optional)"
                )
            
            # Log warnings for items that failed to parse but don't fail the entire operation
            if parse_errors and len(stories) > 0:
                print(f"Warning: {len(parse_errors)} items failed to parse but {len(stories)} succeeded")
                    
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON format at line {e.lineno}, column {e.colno}: {e.msg}\n\n"
                "Please check your JSON syntax. Common issues:\n"
                "- Missing or extra commas\n"
                "- Unclosed brackets or braces\n"
                "- Unquoted strings\n"
                "- Single quotes instead of double quotes"
            )
            
        return stories
    
    def parse_csv(self, csv_data: str) -> List[UserStory]:
        """
        Parse user stories from CSV data.
        
        Args:
            csv_data: CSV string containing work items
            
        Returns:
            List of UserStory objects
        """
        stories = []
        parse_errors = []
        
        try:
            reader = csv.DictReader(StringIO(csv_data))
            
            for idx, row in enumerate(reader):
                try:
                    story = self._parse_csv_row(row)
                    if story:
                        stories.append(story)
                    else:
                        parse_errors.append(f"Row {idx + 2}: Could not parse CSV row (missing required fields)")
                except Exception as e:
                    parse_errors.append(f"Row {idx + 2}: {str(e)}")
            
            if not stories:
                error_details = "\n".join(parse_errors) if parse_errors else "No valid rows found"
                raise ValueError(
                    "Failed to parse any user stories from CSV.\n\n"
                    "Parsing errors:\n"
                    f"{error_details}\n\n"
                    "CSV must have columns: ID, Title (required). Optional: Description, Acceptance Criteria, Priority, Tags"
                )
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"CSV parsing error: {str(e)}")
                
        return stories
    
    def _parse_work_item(self, item: Dict[str, Any]) -> Optional[UserStory]:
        """
        Parse a single work item from JSON into a UserStory.
        
        Args:
            item: Dictionary representing a work item
            
        Returns:
            UserStory object or None if parsing fails
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        if not isinstance(item, dict):
            raise ValueError(f"Work item must be a dictionary, got {type(item).__name__}")
        
        fields = item.get('fields', item)
        
        # Extract basic fields with more lenient handling
        # Handle id as various types (int, str, etc.)
        raw_id = item.get('id', fields.get('System.Id', fields.get('id', None)))
        story_id = str(raw_id) if raw_id is not None else ''
        
        # Handle title with fallback
        title = fields.get('System.Title', fields.get('title', ''))
        if not title:
            title = fields.get('name', '')  # Additional fallback
        
        description = fields.get('System.Description', fields.get('description', ''))
        
        # Validate required fields - be lenient with whitespace-only values
        if not story_id:
            raise ValueError("Missing required field 'id'. Each work item must have an 'id' field.")
        
        if not title:
            raise ValueError(f"Work item '{story_id}' is missing required field 'title'. Each work item must have a 'title' field.")
        
        # Extract acceptance criteria
        ac_raw = fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', 
                          fields.get('acceptance_criteria', fields.get('acceptanceCriteria', '')))
        acceptance_criteria = self._extract_acceptance_criteria(ac_raw)
        
        # Extract persona
        persona = fields.get('persona', self._extract_persona(description))
        
        # Extract priority
        priority = fields.get('Microsoft.VSTS.Common.Priority', 
                            fields.get('priority', 3))
        
        # Extract non-functional hints
        nf_hints = self._extract_non_functional_hints(description, ac_raw)
        
        # Extract tags
        tags = fields.get('System.Tags', fields.get('tags', ''))
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(';') if t.strip()]
        elif isinstance(tags, list):
            tags = [str(t).strip() for t in tags if str(t).strip()]
        else:
            tags = []
        
        return UserStory(
            id=story_id,
            title=title,
            description=self._clean_html(description),
            acceptance_criteria=acceptance_criteria,
            persona=persona,
            priority=int(priority) if priority else 3,
            non_functional_hints=nf_hints,
            tags=tags
        )
    
    def _parse_csv_row(self, row: Dict[str, str]) -> Optional[UserStory]:
        """
        Parse a CSV row into a UserStory.
        
        Args:
            row: Dictionary representing a CSV row
            
        Returns:
            UserStory object or None if parsing fails
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Map common CSV column names - be more lenient
        story_id = row.get('ID', row.get('Work Item ID', row.get('id', row.get('Id', ''))))
        title = row.get('Title', row.get('title', row.get('Name', row.get('name', ''))))
        description = row.get('Description', row.get('description', ''))
        ac_raw = row.get('Acceptance Criteria', row.get('acceptanceCriteria', row.get('acceptance_criteria', '')))
        
        # Validate required fields - more lenient, just check if they exist
        if not story_id:
            raise ValueError("Missing required column 'ID'. Each CSV row must have an 'ID' column.")
        
        if not title:
            raise ValueError(f"Row with ID '{story_id}' is missing required column 'Title'. Each CSV row must have a 'Title' column.")
        
        return UserStory(
            id=str(story_id),
            title=title,
            description=self._clean_html(description),
            acceptance_criteria=self._extract_acceptance_criteria(ac_raw),
            persona=row.get('Persona', self._extract_persona(description)),
            priority=int(row.get('Priority', 3)) if row.get('Priority') else 3,
            non_functional_hints=self._extract_non_functional_hints(description, ac_raw),
            tags=[t.strip() for t in row.get('Tags', '').split(';') if t.strip()]
        )
    
    def _extract_acceptance_criteria(self, ac_raw: Union[str, List[str], Any]) -> List[str]:
        """
        Extract acceptance criteria as a list of strings.
        
        Args:
            ac_raw: Raw acceptance criteria (string or list)
            
        Returns:
            List of acceptance criteria strings
        """
        if not ac_raw:
            return []
        
        # If already a list, return it (cleaning each item)
        if isinstance(ac_raw, list):
            return [str(item).strip() for item in ac_raw if str(item).strip()]
        
        # Convert to string and clean HTML
        clean_text = self._clean_html(str(ac_raw))
        
        # Split by common patterns
        criteria = []
        
        # Try numbered list (1. or 1))
        numbered = re.split(r'\d+[.)]\s*', clean_text)
        if len(numbered) > 1:
            criteria = [c.strip() for c in numbered if c.strip()]
        else:
            # Try bullet points
            bullets = re.split(r'[-•*]\s*', clean_text)
            if len(bullets) > 1:
                criteria = [c.strip() for c in bullets if c.strip()]
            else:
                # Try newlines
                lines = clean_text.split('\n')
                criteria = [l.strip() for l in lines if l.strip()]
        
        # Filter out very short criteria
        return [c for c in criteria if len(c) > 3]
    
    def _extract_persona(self, description: str) -> Optional[str]:
        """
        Extract persona from description using 'As a...' pattern.
        
        Args:
            description: Story description text
            
        Returns:
            Extracted persona or None
        """
        match = re.search(r'[Aa]s\s+(?:a|an)\s+([^,]+)', description)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_non_functional_hints(self, description: str, ac: str) -> List[str]:
        """
        Extract non-functional requirement hints from text.
        
        Args:
            description: Story description
            ac: Acceptance criteria text
            
        Returns:
            List of non-functional hints
        """
        hints = []
        combined = f"{description} {ac}".lower()
        
        # Performance hints
        if any(word in combined for word in ['performance', 'fast', 'quick', 'speed', 'millisecond', 'response time']):
            hints.append('performance')
        
        # Security hints
        if any(word in combined for word in ['secure', 'security', 'authentication', 'authorization', 'encrypt', 'password']):
            hints.append('security')
        
        # Scalability hints
        if any(word in combined for word in ['scale', 'scalable', 'concurrent', 'load', 'traffic']):
            hints.append('scalability')
        
        # Accessibility hints
        if any(word in combined for word in ['accessible', 'accessibility', 'a11y', 'wcag', 'screen reader']):
            hints.append('accessibility')
        
        # Reliability hints
        if any(word in combined for word in ['reliable', 'reliability', 'uptime', 'available', 'fault tolerant']):
            hints.append('reliability')
        
        return hints
    
    def _clean_html(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Text potentially containing HTML
            
        Returns:
            Clean text without HTML tags
        """
        if not text:
            return ''
        
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', ' ', str(text))
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
    
    def normalize_to_spec(self, stories: List[UserStory], 
                         tech_stack: Optional[Dict[str, str]] = None,
                         constraints: Optional[Dict[str, Any]] = None) -> CanonicalSpec:
        """
        Normalize user stories into a canonical specification.
        
        Args:
            stories: List of UserStory objects
            tech_stack: Optional technology stack specification
            constraints: Optional project constraints
            
        Returns:
            CanonicalSpec object for downstream agents
        """
        # Derive requirements from stories
        requirements = {
            'functional': [],
            'non_functional': set(),
            'personas': set(),
            'features': []
        }
        
        for story in stories:
            # Collect functional requirements
            requirements['functional'].extend([
                {'story_id': story.id, 'criterion': ac}
                for ac in story.acceptance_criteria
            ])
            
            # Collect non-functional requirements
            requirements['non_functional'].update(story.non_functional_hints)
            
            # Collect personas
            if story.persona:
                requirements['personas'].add(story.persona)
            
            # Collect feature tags
            requirements['features'].extend(story.tags)
        
        # Convert sets to lists for JSON serialization
        requirements['non_functional'] = list(requirements['non_functional'])
        requirements['personas'] = list(requirements['personas'])
        requirements['features'] = list(set(requirements['features']))
        
        # Default tech stack if not provided
        default_tech_stack = {
            'frontend': 'React',
            'backend': 'FastAPI',
            'database': 'PostgreSQL',
            'testing': 'pytest'
        }
        
        return CanonicalSpec(
            user_stories=stories,
            requirements=requirements,
            tech_stack=tech_stack or default_tech_stack,
            constraints=constraints or {}
        )
    
    def fetch_work_items_from_ado(self, query: Optional[str] = None, 
                                   work_item_ids: Optional[List[str]] = None) -> List[UserStory]:
        """
        Fetch work items directly from Azure DevOps REST API.
        
        Args:
            query: Optional WIQL query to filter work items
            work_item_ids: Optional list of specific work item IDs to fetch
            
        Returns:
            List of UserStory objects
            
        Raises:
            ValueError: If credentials are missing or API call fails
        """
        if not requests:
            raise ValueError(
                "The 'requests' library is required for Azure DevOps API integration.\n"
                "Install it with: pip install requests"
            )
        
        if not self.ado_url or not self.pat or not self.project:
            raise ValueError(
                "Azure DevOps credentials are required. Please set:\n"
                "- ADO_ORG_URL (e.g., https://dev.azure.com/your-org)\n"
                "- ADO_PAT (Personal Access Token)\n"
                "- ADO_PROJECT (Project name)"
            )
        
        # Create authorization header
        auth_string = f":{self.pat}"
        auth_bytes = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
        headers = {
            'Authorization': f'Basic {auth_bytes}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        try:
            # If specific IDs are provided, fetch them directly
            if work_item_ids:
                ids_str = ','.join(work_item_ids)
                url = f"{self.ado_url}/{self.project}/_apis/wit/workitems?ids={ids_str}&api-version=7.0"
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                items = data.get('value', [])
            else:
                # Otherwise, use a query (default to all User Stories)
                if not query:
                    query = f"SELECT [System.Id] FROM WorkItems WHERE [System.WorkItemType] = 'User Story' AND [System.TeamProject] = '{self.project}'"
                
                # Execute WIQL query
                query_url = f"{self.ado_url}/{self.project}/_apis/wit/wiql?api-version=7.0"
                query_payload = {"query": query}
                response = requests.post(query_url, json=query_payload, headers=headers, timeout=30)
                response.raise_for_status()
                query_result = response.json()
                
                # Extract work item IDs from query result
                work_items = query_result.get('workItems', [])
                if not work_items:
                    return []
                
                ids = [str(item['id']) for item in work_items]
                ids_str = ','.join(ids)
                
                # Fetch full work item details
                details_url = f"{self.ado_url}/{self.project}/_apis/wit/workitems?ids={ids_str}&api-version=7.0"
                response = requests.get(details_url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                items = data.get('value', [])
            
            # Parse the fetched items
            stories = []
            for item in items:
                try:
                    story = self._parse_work_item(item)
                    if story:
                        stories.append(story)
                except Exception as e:
                    # Log but don't fail on individual item errors
                    print(f"Warning: Failed to parse work item {item.get('id', 'unknown')}: {e}")
            
            return stories
            
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Failed to connect to Azure DevOps: {str(e)}\n\n"
                "Please verify:\n"
                "- Organization URL is correct\n"
                "- Personal Access Token has 'Work Items (Read)' permission\n"
                "- Project name is correct\n"
                "- You have network access to Azure DevOps"
            )
    
    def commit_code_to_ado_repo(
        self,
        artifacts: list,
        repo_name: str,
        branch: str = "refs/heads/generated-code",
        commit_message: str = "Add generated code from Agentic Code Generator"
    ) -> dict:
        """
        Commit generated code artifacts to an Azure DevOps Git repository.

        Args:
            artifacts: List of GeneratedArtifact objects to commit
            repo_name: Name of the Azure Repos repository
            branch: Branch to commit to
            commit_message: Commit message

        Returns:
            Dict with commit info
        """
        import base64
        import requests

        print("endpoint hit")

        if not self.ado_url or not self.pat or not self.project:
            raise ValueError("Azure DevOps credentials are required for code commits")

        if not artifacts:
            raise ValueError("No artifacts to commit")

        # Deduplicate artifacts by file_path
        unique_artifacts = {}
        for artifact in artifacts:
            file_path = getattr(artifact, "file_path", "").strip()
            if not file_path:
                continue
            unique_artifacts[file_path] = artifact  # last one wins

        artifacts = list(unique_artifacts.values())
        print(f"[ADO] {len(artifacts)} unique artifacts after deduplication")

        print(f"[ADO] Committing {len(artifacts)} validated artifacts to {repo_name} on {branch}")

        # Authorization header
        auth_string = f":{self.pat}"
        auth_bytes = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
        headers = {
            'Authorization': f'Basic {auth_bytes}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        try:
            # Get repository info
            repo_url = f"{self.ado_url}/{self.project}/_apis/git/repositories/{repo_name}?api-version=7.0"
            print(f"[ADO] Fetching repository info from: {repo_url}")
            response = requests.get(repo_url, headers=headers, timeout=30)
            response.raise_for_status()
            repo_data = response.json()
            repo_id = repo_data['id']
            print(f"[ADO] Repository ID: {repo_id}")

            # Resolve branch commit
            branch_ref = branch.replace("refs/heads/", "")
            refs_url = f"{self.ado_url}/{self.project}/_apis/git/repositories/{repo_id}/refs?filter=heads/{branch_ref}&api-version=7.0"
            response = requests.get(refs_url, headers=headers, timeout=30)
            response.raise_for_status()
            refs_data = response.json()
            if refs_data.get('value'):
                old_object_id = refs_data['value'][0]['objectId']
                branch_exists = True
                print(f"[ADO] Branch exists with commit: {old_object_id}")
            else:
                # Branch doesn't exist, use default branch
                default_branch_ref = repo_data['defaultBranch'].replace("refs/heads/", "")
                refs_url = f"{self.ado_url}/{self.project}/_apis/git/repositories/{repo_id}/refs?filter=heads/{default_branch_ref}&api-version=7.0"
                response = requests.get(refs_url, headers=headers, timeout=30)
                response.raise_for_status()
                refs_data = response.json()
                old_object_id = refs_data['value'][0]['objectId'] if refs_data.get('value') else None
                branch_exists = False
                if old_object_id:
                    print(f"[ADO] Default branch commit: {old_object_id}")
                else:
                    old_object_id = "0000000000000000000000000000000000000000"

            # Build changes
            changes = []
            for i, artifact in enumerate(artifacts):
                file_path = getattr(artifact, "file_path", "").strip()
                content = getattr(artifact, "content", "").strip()
                if not file_path.startswith('/'):
                    file_path = '/' + file_path

                changes.append({
                    "changeType": "add",
                    "item": {"path": file_path},
                    "newContent": {"content": content, "contentType": "rawtext"}
                })
                print(f"[ADO] Adding file {i+1}/{len(artifacts)}: {file_path} ({len(content)} chars)")

            # Build push payload
            push_payload = {
                "refUpdates": [{"name": branch, "oldObjectId": old_object_id}],
                "commits": [{"comment": commit_message, "changes": changes}]
            }

            print(f"[ADO] Push payload ready: {len(changes)} file(s), branch {branch}, oldObjectId {old_object_id}")

            # Push
            push_url = f"{self.ado_url}/{self.project}/_apis/git/repositories/{repo_id}/pushes?api-version=7.0"
            response = requests.post(push_url, json=push_payload, headers=headers, timeout=60)

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                status_code = getattr(e.response, "status_code", "unknown")
                error_text = getattr(e.response, "text", str(e))
                print(f"[ADO] Push failed with status {status_code}\n{error_text}")
                raise ValueError(f"Failed to commit code to Azure DevOps (Status {status_code}).\n{error_text}")

            push_result = response.json()
            commit_id = push_result.get('commits', [{}])[0].get('commitId')

            print(f"[ADO] ✓ Successfully pushed {len(changes)} files, commit: {commit_id}")

            return {
                "success": True,
                "commit_id": commit_id,
                "branch": branch,
                "repository": repo_name,
                "files_committed": len(changes),
                "url": push_result.get('url', ''),
                "push_id": push_result.get('pushId')
            }

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to connect to Azure DevOps: {str(e)}")

    def create_test_plan(self, test_plan: 'TestPlan') -> Dict[str, Any]:
        """
        Create a test plan in Azure DevOps.
        
        Args:
            test_plan: TestPlan object with test suites and cases
            
        Returns:
            Dict with test plan creation result
            
        Raises:
            ValueError: If credentials are missing or creation fails
        """
        if not requests:
            raise ValueError(
                "The 'requests' library is required for Azure DevOps API integration.\n"
                "Install it with: pip install requests"
            )
        
        if not self.ado_url or not self.pat or not self.project:
            raise ValueError("Azure DevOps credentials are required for test plan creation")
        
        print(f"[ADO] Creating test plan: {test_plan.name}")
        
        # Create authorization header with Azure DevOps required headers
        auth_string = f":{self.pat}"
        auth_bytes = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
        headers = {
            'Authorization': f'Basic {auth_bytes}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        try:
            # Create test plan
            plan_url = f"{self.ado_url}/{self.project}/_apis/testplan/plans?api-version=7.0"
            
            plan_payload = {
                "name": test_plan.name,
                "description": test_plan.description,
                "areaPath": test_plan.area_path or self.project,
                "iteration": test_plan.iteration or self.project
            }
            
            print(f"[ADO] Creating test plan at: {plan_url}")
            response = requests.post(plan_url, json=plan_payload, headers=headers, timeout=30)
            
            if response.status_code not in [200, 201]:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = json.dumps(error_data, indent=2)
                except:
                    error_detail = response.text
                
                print(f"[ADO] Test plan creation failed with status {response.status_code}")
                print(f"[ADO] Error response: {error_detail}")
                
                raise requests.exceptions.HTTPError(
                    f"Azure DevOps returned status {response.status_code}. "
                    f"Response: {error_detail}"
                )
            
            plan_result = response.json()
            plan_id = plan_result.get('id')
            print(f"[ADO] ✓ Test plan created with ID: {plan_id}")
            
            # Create test suites for each suite in the plan
            suite_results = []
            for suite in test_plan.test_suites:
                suite_url = f"{self.ado_url}/{self.project}/_apis/testplan/plans/{plan_id}/suites?api-version=7.0"
                
                suite_payload = {
                    "name": suite.name,
                    "suiteType": "StaticTestSuite",
                    "parentSuite": {
                        "id": plan_result.get('rootSuite', {}).get('id')
                    }
                }
                
                print(f"[ADO] Creating test suite: {suite.name}")
                response = requests.post(suite_url, json=suite_payload, headers=headers, timeout=30)
                
                if response.status_code in [200, 201]:
                    suite_result = response.json()
                    suite_id = suite_result.get('id')
                    print(f"[ADO] ✓ Test suite created with ID: {suite_id}")
                    
                    # Create test cases (simplified - actual implementation would be more complex)
                    test_case_ids = []
                    for test_case in suite.test_cases:
                        print(f"[ADO]   - Test case: {test_case.title} (automation: {test_case.automated})")
                        test_case_ids.append(test_case.id)
                    
                    suite_results.append({
                        "suite_id": suite_id,
                        "name": suite.name,
                        "test_case_count": len(suite.test_cases)
                    })
                else:
                    print(f"[ADO] Warning: Failed to create test suite {suite.name}")
            
            return {
                "success": True,
                "plan_id": plan_id,
                "plan_name": test_plan.name,
                "url": plan_result.get('url', ''),
                "suites_created": len(suite_results),
                "suites": suite_results
            }
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
            
            raise ValueError(
                f"Failed to create test plan in Azure DevOps (Status {status_code}): {error_msg}\n\n"
                "Common issues:\n"
                "- Personal Access Token missing 'Test Management (Read & Write)' permission\n"
                "- No permission to create test plans in the project\n"
                "- Invalid project name or area path"
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Failed to connect to Azure DevOps: {str(e)}\n\n"
                "Please verify:\n"
                "- Organization URL is correct\n"
                "- Network connectivity to Azure DevOps\n"
                "- PAT is valid and not expired"
            )
    
    def publish_to_azure_devops(self, 
                                pipeline_state: 'PipelineState',
                                repo_name: str,
                                branch: str = "refs/heads/generated-code") -> Dict[str, Any]:
        """
        Publish all pipeline artifacts to their respective Azure DevOps sections.
        
        This method categorizes artifacts and publishes them to:
        - Azure Repos: Code artifacts (source code, configs)
        - Azure Test Plans: Test plans with test suites and cases
        - Azure Boards: Work items for tracking
        - Azure Artifacts: Build outputs (via pipeline)
        
        Args:
            pipeline_state: Complete pipeline state with all artifacts
            repo_name: Target repository name
            branch: Target branch
            
        Returns:
            Dict with results from each Azure section
        """
        results = {
            "success": True,
            "repos": None,
            "test_plans": None,
            "boards": None,
            "errors": []
        }
        
        print(f"[ADO] Publishing pipeline artifacts to Azure DevOps")
        print(f"[ADO] Total artifacts: {len(pipeline_state.artifacts)}")
        print(f"[ADO] Test plans: {len(pipeline_state.test_plans)}")
        
        # 1. Categorize artifacts
        code_artifacts = []
        test_artifacts = []
        config_artifacts = []
        doc_artifacts = []
        
        for artifact in pipeline_state.artifacts:
            artifact_dict = artifact.to_dict() if hasattr(artifact, 'to_dict') else artifact
            
            # Categorize based on file path and type
            file_path = artifact_dict.get('file_path', '')
            artifact_type = artifact_dict.get('artifact_type', 'code')
            
            if 'test' in file_path or artifact_type == 'test':
                test_artifacts.append(artifact_dict)
            elif artifact_type == 'config' or file_path.endswith(('.yml', '.yaml', '.json', '.toml')):
                config_artifacts.append(artifact_dict)
            elif artifact_type == 'documentation' or file_path.endswith(('.md', '.txt', '.rst')):
                doc_artifacts.append(artifact_dict)
            else:
                code_artifacts.append(artifact_dict)
        
        print(f"[ADO] Categorized artifacts:")
        print(f"[ADO]   - Code: {len(code_artifacts)}")
        print(f"[ADO]   - Tests: {len(test_artifacts)}")
        print(f"[ADO]   - Config: {len(config_artifacts)}")
        print(f"[ADO]   - Documentation: {len(doc_artifacts)}")
        
        # 2. Commit code artifacts to Azure Repos
        try:
            all_artifacts_for_repo = code_artifacts + test_artifacts + config_artifacts + doc_artifacts
            
            if all_artifacts_for_repo:
                print(f"\n[ADO] ======================================")
                print(f"[ADO] 📦 AZURE REPOS: Committing Code")
                print(f"[ADO] ======================================")
                
                commit_result = self.commit_code_to_ado_repo(
                    artifacts=all_artifacts_for_repo,
                    repo_name=repo_name,
                    branch=branch,
                    commit_message=f"Generated code with {len(all_artifacts_for_repo)} artifacts"
                )
                results['repos'] = commit_result
                print(f"[ADO] ✓ Code committed to Azure Repos: {commit_result.get('commit_id', 'N/A')}")
            else:
                print(f"[ADO] ⚠ No code artifacts to commit to Azure Repos")
        except Exception as e:
            error_msg = f"Failed to commit to Azure Repos: {str(e)}"
            print(f"[ADO] ✗ {error_msg}")
            results['errors'].append(error_msg)
            results['success'] = False
        
        # 3. Create test plans in Azure Test Plans
        try:
            if pipeline_state.test_plans:
                print(f"\n[ADO] ======================================")
                print(f"[ADO] 🧪 AZURE TEST PLANS: Creating Test Plans")
                print(f"[ADO] ======================================")
                
                test_plan_results = []
                for test_plan in pipeline_state.test_plans:
                    # Use test_plan object directly - it's already the right type
                    plan_result = self.create_test_plan(test_plan)
                    test_plan_results.append(plan_result)
                    print(f"[ADO] ✓ Test plan created: {plan_result.get('plan_name', 'N/A')} (ID: {plan_result.get('plan_id', 'N/A')})")
                
                results['test_plans'] = test_plan_results
            else:
                print(f"[ADO] ⚠ No test plans to create in Azure Test Plans")
        except Exception as e:
            error_msg = f"Failed to create test plans: {str(e)}"
            print(f"[ADO] ✗ {error_msg}")
            results['errors'].append(error_msg)
            # Don't fail the whole operation if test plans fail
        
        # 4. Create work items in Azure Boards for tracking
        try:
            if pipeline_state.spec and pipeline_state.spec.user_stories:
                print(f"\n[ADO] ======================================")
                print(f"[ADO] 📋 AZURE BOARDS: Creating Work Items")
                print(f"[ADO] ======================================")
                
                board_results = self._create_tracking_work_items(pipeline_state)
                results['boards'] = board_results
            else:
                print(f"[ADO] ⚠ No user stories to create work items in Azure Boards")
        except Exception as e:
            error_msg = f"Failed to create work items: {str(e)}"
            print(f"[ADO] ✗ {error_msg}")
            results['errors'].append(error_msg)
            # Don't fail the whole operation if boards fail
        
        # 5. Summary
        print(f"\n[ADO] ======================================")
        print(f"[ADO] 📊 PUBLICATION SUMMARY")
        print(f"[ADO] ======================================")
        print(f"[ADO] Azure Repos: {'✓ Success' if results['repos'] else '✗ Failed'}")
        print(f"[ADO] Azure Test Plans: {'✓ Success' if results['test_plans'] else '⚠ Skipped'}")
        print(f"[ADO] Azure Boards: {'✓ Success' if results['boards'] else '⚠ Skipped'}")
        print(f"[ADO] Azure Artifacts: 📦 Configured via azure-pipelines.yml")
        
        if results['errors']:
            print(f"[ADO] ⚠ {len(results['errors'])} error(s) occurred:")
            for error in results['errors']:
                print(f"[ADO]   - {error}")
        
        return results
    
    def _create_tracking_work_items(self, pipeline_state: 'PipelineState') -> Dict[str, Any]:
        """
        Create work items in Azure Boards for tracking generated code.
        
        Args:
            pipeline_state: Pipeline state with spec and artifacts
            
        Returns:
            Dict with work item creation results
        """
        if not requests:
            raise ValueError("The 'requests' library is required for Azure DevOps API integration.")
        
        if not self.ado_url or not self.pat or not self.project:
            raise ValueError("Azure DevOps credentials are required")
        
        # Create authorization header with Azure DevOps required headers
        # Note: Work Items API uses application/json-patch+json for PATCH operations
        auth_string = f":{self.pat}"
        auth_bytes = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
        headers = {
            'Authorization': f'Basic {auth_bytes}',
            'Content-Type': 'application/json-patch+json',
            'Accept': 'application/json'
        }
        
        work_items_created = []
        
        try:
            # Create a parent Epic for the generated code
            epic_url = f"{self.ado_url}/{self.project}/_apis/wit/workitems/$Epic?api-version=7.0"
            
            epic_payload = [
                {
                    "op": "add",
                    "path": "/fields/System.Title",
                    "value": f"Generated Code - {pipeline_state.spec.project_name or 'Project'}"
                },
                {
                    "op": "add",
                    "path": "/fields/System.Description",
                    "value": f"Automated code generation completed with {len(pipeline_state.artifacts)} artifacts"
                },
                {
                    "op": "add",
                    "path": "/fields/System.Tags",
                    "value": "auto-generated;agentic-code"
                }
            ]
            
            response = requests.post(epic_url, json=epic_payload, headers=headers, timeout=30)
            
            if response.status_code in [200, 201]:
                epic_data = response.json()
                epic_id = epic_data.get('id')
                print(f"[ADO] ✓ Created Epic work item: {epic_id}")
                work_items_created.append({
                    "type": "Epic",
                    "id": epic_id,
                    "title": epic_data.get('fields', {}).get('System.Title')
                })
                
                # Create Feature work items for each major component
                components = set()
                for artifact in pipeline_state.artifacts:
                    path = artifact.file_path if hasattr(artifact, 'file_path') else artifact.get('file_path', '')
                    if '/' in path:
                        component = path.split('/')[1] if path.startswith('/') else path.split('/')[0]
                        components.add(component)
                
                for component in list(components)[:MAX_WORK_ITEMS_PER_COMPONENT]:
                    feature_url = f"{self.ado_url}/{self.project}/_apis/wit/workitems/$Feature?api-version=7.0"
                    
                    feature_payload = [
                        {
                            "op": "add",
                            "path": "/fields/System.Title",
                            "value": f"Generated Component: {component}"
                        },
                        {
                            "op": "add",
                            "path": "/fields/System.Parent",
                            "value": epic_id
                        },
                        {
                            "op": "add",
                            "path": "/fields/System.Tags",
                            "value": "auto-generated"
                        }
                    ]
                    
                    response = requests.post(feature_url, json=feature_payload, headers=headers, timeout=30)
                    
                    if response.status_code in [200, 201]:
                        feature_data = response.json()
                        feature_id = feature_data.get('id')
                        print(f"[ADO] ✓ Created Feature work item: {feature_id} ({component})")
                        work_items_created.append({
                            "type": "Feature",
                            "id": feature_id,
                            "title": component
                        })
            
            return {
                "success": True,
                "work_items_created": len(work_items_created),
                "work_items": work_items_created
            }
            
        except Exception as e:
            print(f"[ADO] Warning: Failed to create some work items: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "work_items_created": len(work_items_created),
                "work_items": work_items_created
            }
    
    def process(self, data: str, data_format: str = 'json',
               tech_stack: Optional[Dict[str, str]] = None,
               constraints: Optional[Dict[str, Any]] = None) -> CanonicalSpec:
        """
        Main entry point to process ADO data.
        
        Args:
            data: Raw data string (JSON or CSV)
            data_format: Format of input data ('json' or 'csv')
            tech_stack: Optional technology stack
            constraints: Optional constraints
            
        Returns:
            CanonicalSpec for downstream agents
        """
        # Parse based on format
        if data_format.lower() == 'csv':
            stories = self.parse_csv(data)
        else:
            stories = self.parse_json(data)
        
        # Normalize to canonical spec
        return self.normalize_to_spec(stories, tech_stack, constraints)
