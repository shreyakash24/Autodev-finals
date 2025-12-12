"""
Flask Application for the Agentic Code Generation System
Provides REST API and WebSocket for real-time updates
"""

import os
import json
import io
import zipfile
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, render_template, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from ..models import (
    CanonicalSpec, TaskStatus, PipelineState,
    UserStory, GeneratedArtifact, TestReport
)
from ..agents import (
    ADOConnectorAgent, OrchestratorAgent,
    FrontendCodingAgent, BackendCodingAgent, DatabaseCodingAgent,
    TestingAgent, LegacyAnalyzerAgent, PromptRefinerAgent, MonitoringAgent
)


# Global state for pipeline (shared across all requests)
current_pipeline = {"state": None}

# Global agent instances (shared across all requests)
ado_agent = None
orchestrator = None
frontend_agent = None
backend_agent = None
database_agent = None
testing_agent = None
legacy_agent = None
prompt_agent = None
monitoring_agent = None
basedir = os.path.abspath(os.path.dirname(__file__))
    

def create_app():
    """Create and configure the Flask application."""
    global ado_agent, orchestrator, frontend_agent, backend_agent, database_agent
    global testing_agent, legacy_agent, prompt_agent, monitoring_agent
    
    app = Flask(__name__, static_folder=os.path.join(basedir, 'static'),
        template_folder=os.path.join(basedir, 'templates'))
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # Flask 3.x configuration: ensure JSON responses use proper Content-Type
    # This prevents issues with Azure DevOps and other strict API consumers
    app.config['JSONIFY_MIMETYPE'] = 'application/json'
    
    # Enable CORS
    CORS(app, origins="*")
    
    # Initialize SocketIO for real-time updates
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    
    # Initialize agents
    ado_agent = ADOConnectorAgent()
    orchestrator = OrchestratorAgent()
    frontend_agent = FrontendCodingAgent()
    backend_agent = BackendCodingAgent()
    database_agent = DatabaseCodingAgent()
    testing_agent = TestingAgent()
    legacy_agent = LegacyAnalyzerAgent()
    prompt_agent = PromptRefinerAgent()
    monitoring_agent = MonitoringAgent()
    
    # Register monitoring callback for WebSocket broadcasts
    def broadcast_event(event):
        socketio.emit('pipeline_event', event.to_dict())
    
    monitoring_agent.subscribe(broadcast_event)
    
    # ============== REST API Routes ==============
    
    @app.route('/', methods=['GET'])
    def index():
        """Serve the main dashboard."""
        return render_template('index.html')
    
    @app.route('/api/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/debug/pipeline')
    def debug_pipeline():
        """Debug endpoint to check pipeline state."""
        if not current_pipeline["state"]:
            return jsonify({"error": "No pipeline state"})
        
        return jsonify({
            "has_state": current_pipeline["state"] is not None,
            "pipeline_id": current_pipeline["state"].id if current_pipeline["state"] else None,
            "task_count": len(current_pipeline["state"].tasks) if current_pipeline["state"] else 0,
            "orchestrator_has_pipeline": orchestrator.pipeline_state is not None,
            "same_reference": orchestrator.pipeline_state == current_pipeline["state"]
        })
    
    # -------------- ADO Connector Routes --------------
    
    @app.route('/api/ado/parse', methods=['POST'])
    def parse_ado_data():
        """
        Parse ADO data (JSON or CSV) and extract user stories.
        
        Request body:
        {
            "data": "<json or csv string>",
            "format": "json" | "csv",
            "tech_stack": {...} (optional),
            "constraints": {...} (optional)
        }
        """
        try:
            body = request.get_json(force=True, silent=False)
            
            if not body:
                response = jsonify({
                    "success": False,
                    "error": "No JSON body provided"
                })
                response.headers['Content-Type'] = 'application/json'
                return response, 400
            
            data = body.get('data', '')
            data_format = body.get('format', 'json')
            tech_stack = body.get('tech_stack')
            constraints = body.get('constraints')
            
            spec = ado_agent.process(data, data_format, tech_stack, constraints)
            
            response = jsonify({
                "success": True,
                "spec": spec.to_dict()
            })
            response.headers['Content-Type'] = 'application/json'
            return response
            
        except Exception as e:
            app.logger.error(f"[Parse] Error: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            response = jsonify({
                "success": False,
                "error": str(e)
            })
            response.headers['Content-Type'] = 'application/json'
            return response, 400
    
    @app.route('/api/ado/fetch', methods=['POST'])
    def fetch_from_ado():
        """
        Fetch work items directly from Azure DevOps.
        
        Request body:
        {
            "org_url": "https://dev.azure.com/your-org",
            "pat": "your_personal_access_token",
            "project": "your_project_name",
            "query": "optional WIQL query",
            "work_item_ids": ["optional", "list", "of", "ids"]
        }
        """
        try:
            body = request.get_json(force=True, silent=False)
            
            if not body:
                response = jsonify({
                    "success": False,
                    "error": "No JSON body provided"
                })
                response.headers['Content-Type'] = 'application/json'
                return response, 400
            
            org_url = body.get('org_url')
            pat = body.get('pat')
            project = body.get('project')
            query = body.get('query')
            work_item_ids = body.get('work_item_ids')
            
            if not org_url or not pat or not project:
                response = jsonify({
                    "success": False,
                    "error": "Missing required parameters: org_url, pat, and project"
                })
                response.headers['Content-Type'] = 'application/json'
                return response, 400
            
            # Create a temporary ADO agent with provided credentials
            from ..agents.ado_connector import ADOConnectorAgent
            temp_agent = ADOConnectorAgent(ado_url=org_url, pat=pat)
            temp_agent.project = project
            
            # Fetch work items
            app.logger.info(f"[ADO] Fetching work items from {org_url}/{project}")
            stories = temp_agent.fetch_work_items_from_ado(query=query, work_item_ids=work_item_ids)
            
            # Normalize to canonical spec
            spec = temp_agent.normalize_to_spec(stories)
            
            app.logger.info(f"[ADO] ✓ Fetched {len(stories)} work items")
            
            response = jsonify({
                "success": True,
                "spec": spec.to_dict(),
                "fetched_count": len(stories)
            })
            response.headers['Content-Type'] = 'application/json'
            return response
            
        except Exception as e:
            app.logger.error(f"[ADO] Fetch error: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            response = jsonify({
                "success": False,
                "error": str(e)
            })
            response.headers['Content-Type'] = 'application/json'
            return response, 400
    
    # Azure DevOps commit endpoints removed - artifacts are now displayed and exported instead
    # @app.route('/api/ado/commit', methods=['POST'])
    # def commit_to_ado():
    #     # REMOVED: This endpoint was used to commit artifacts to Azure DevOps
    #     # Now artifacts are displayed on UI and can be exported
    #     pass
    
    # Azure DevOps commit-current endpoint removed - artifacts are now displayed and exported instead
    # @app.route('/api/ado/commit-current', methods=['POST'])
    # def commit_current_artifacts():
    #     # REMOVED: This endpoint was used to commit current pipeline artifacts to Azure DevOps
    #     # Now artifacts are displayed on UI and can be exported
    #     pass
    
    # -------------- Pipeline Routes --------------
    
    @app.route('/api/pipeline/create', methods=['POST'])
    def create_pipeline():
        """
        Create a new pipeline from canonical spec.
        
        Request body:
        {
            "spec": {...}  // CanonicalSpec as dict
        }
        """
        try:
            # Get JSON with explicit handling for Flask 3.x
            try:
                body = request.get_json(force=True, silent=False)
            except Exception as json_error:
                app.logger.error(f"[Pipeline] JSON parsing error: {json_error}")
                response = jsonify({
                    "success": False,
                    "error": "Invalid JSON in request body"
                })
                response.headers['Content-Type'] = 'application/json'
                return response, 400
            
            if not body:
                app.logger.error("[Pipeline] No JSON body in request")
                response = jsonify({
                    "success": False,
                    "error": "No JSON body provided"
                })
                response.headers['Content-Type'] = 'application/json'
                return response, 400
            
            spec_data = body.get('spec', {})
            
            if not spec_data:
                app.logger.error("[Pipeline] No spec data in request body")
                return jsonify({
                    "success": False,
                    "error": "Missing 'spec' in request body"
                }), 400
            
            app.logger.info(f"[Pipeline] Creating pipeline with {len(spec_data.get('user_stories', []))} user stories")
            
            # Reconstruct spec from dict with error handling
            stories = []
            for s in spec_data.get('user_stories', []):
                try:
                    story = UserStory(
                        id=s.get('id', ''),
                        title=s.get('title', ''),
                        description=s.get('description', ''),
                        acceptance_criteria=s.get('acceptance_criteria', []),
                        persona=s.get('persona'),
                        priority=s.get('priority', 3),
                        non_functional_hints=s.get('non_functional_hints', []),
                        tags=s.get('tags', [])
                    )
                    stories.append(story)
                except (KeyError, TypeError) as e:
                    app.logger.warning(f"[Pipeline] Skipping malformed story: {e}")
                    continue  # Skip malformed stories
            
            if not stories:
                app.logger.warning("[Pipeline] No valid user stories found")
            
            spec = CanonicalSpec(
                user_stories=stories,
                requirements=spec_data.get('requirements', {}),
                tech_stack=spec_data.get('tech_stack', {}),
                constraints=spec_data.get('constraints', {}),
                project_name=spec_data.get('project_name', 'Generated Code')
            )
            
            # Build pipeline
            app.logger.info("[Pipeline] Building pipeline from spec")
            pipeline = orchestrator.build_pipeline(spec)
            current_pipeline["state"] = pipeline
            monitoring_agent.set_pipeline(pipeline)
            
            app.logger.info(f"[Pipeline] ✓ Pipeline created with {len(pipeline.tasks)} tasks")
            
            # Create response without charset in Content-Type
            response = jsonify({
                "success": True,
                "pipeline": pipeline.to_dict()
            })
            response.headers['Content-Type'] = 'application/json'
            return response
            
        except ValueError as e:
            app.logger.error(f"[Pipeline] ValueError: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            response = jsonify({
                "success": False,
                "error": f"Invalid request data: {str(e)}"
            })
            response.headers['Content-Type'] = 'application/json'
            return response, 400
            
        except Exception as e:
            app.logger.error(f"[Pipeline] Unexpected error: {e}")
            import traceback
            app.logger.error(traceback.format_exc())
            response = jsonify({
                "success": False,
                "error": f"Pipeline creation failed: {str(e)}"
            })
            response.headers['Content-Type'] = 'application/json'
            return response, 500
    
    @app.route('/api/pipeline/status', methods=['GET'])
    def get_pipeline_status():
        """Get current pipeline status."""
        if not current_pipeline["state"]:
            return jsonify({
                "success": False,
                "error": "No pipeline active"
            }), 404
        
        return jsonify({
            "success": True,
            "pipeline": current_pipeline["state"].to_dict(),
            "summary": orchestrator.get_pipeline_summary()
        })
    
    @app.route('/api/pipeline/tasks')
    def get_tasks():
        """Get all tasks in the pipeline."""
        if not current_pipeline["state"]:
            return jsonify({"tasks": []})
        
        return jsonify({
            "tasks": [t.to_dict() for t in current_pipeline["state"].tasks]
        })
    
    @app.route('/api/pipeline/ready-tasks')
    def get_ready_tasks():
        """Get tasks that are ready to execute."""
        ready = orchestrator.get_ready_tasks()
        return jsonify({
            "tasks": [t.to_dict() for t in ready]
        })
    
    @app.route('/api/pipeline/dependency-graph')
    def get_dependency_graph():
        """Get the task dependency graph."""
        return jsonify({
            "graph": orchestrator.get_dependency_graph()
        })
    
    @app.route('/api/pipeline/quality-gate')
    def check_quality_gate():
        """Check quality gate status."""
        return jsonify(orchestrator.check_quality_gate())
    
    # -------------- Task Execution Routes --------------
    
    @app.route('/api/task/<task_id>/execute', methods=['POST'])
    def execute_task(task_id):
        """
        Execute a specific task.
        
        This is a simplified execution that generates code based on task type.
        In a full implementation, this would involve CrewAI task execution.
        """
        if not current_pipeline["state"]:
            return jsonify({"success": False, "error": "No pipeline"}), 404
        
        task = next(
            (t for t in current_pipeline["state"].tasks if t.id == task_id),
            None
        )
        
        if not task:
            return jsonify({"success": False, "error": "Task not found"}), 404
        
        try:
            # Update task status
            orchestrator.update_task_status(task_id, TaskStatus.IN_PROGRESS)
            
            # Execute based on agent type
            artifacts = []
            requirements = task.input_data.get('requirements', {})
            
            app.logger.info(f"Executing task {task_id} with agent type {task.agent_type.value}")
            
            if task.agent_type.value == 'frontend_coder':
                artifacts = frontend_agent.generate_component_scaffold(requirements)
            elif task.agent_type.value == 'backend_coder':
                artifacts = backend_agent.generate_api_contracts(requirements)
            elif task.agent_type.value == 'database_coder':
                artifacts = database_agent.generate_schema(requirements)
            elif task.agent_type.value == 'testing':
                test_type = task.input_data.get('test_type', 'unit')
                if test_type == 'unit':
                    artifacts = testing_agent.generate_unit_tests(requirements, [])
                elif test_type == 'integration':
                    artifacts = testing_agent.generate_integration_tests(requirements)
                else:
                    artifacts = testing_agent.generate_e2e_tests(requirements)
            
            app.logger.info(f"Generated {len(artifacts)} artifacts for task {task_id}")
            
            # Validate artifacts are not empty
            if not artifacts:
                app.logger.warning(f"Task {task_id} generated no artifacts")
            else:
                empty_artifacts = []
                for artifact in artifacts:
                    if not artifact.content or artifact.content.strip() == '':
                        empty_artifacts.append(artifact.file_path)
                        app.logger.warning(f"Artifact {artifact.file_path} has empty content")
                
                if empty_artifacts:
                    app.logger.warning(f"Task {task_id} generated {len(empty_artifacts)} empty artifacts: {empty_artifacts}")
            
            # Add artifacts to pipeline
            for artifact in artifacts:
                orchestrator.add_artifact(artifact)
                app.logger.info(f"Added artifact: {artifact.file_path} ({len(artifact.content)} chars)")
            
            # Update task status
            orchestrator.update_task_status(
                task_id, 
                TaskStatus.COMPLETED,
                output_data={"artifacts": [a.id for a in artifacts]}
            )
            
            return jsonify({
                "success": True,
                "artifacts": [a.to_dict() for a in artifacts],
                "artifacts_count": len(artifacts),
                "empty_artifacts_count": len([a for a in artifacts if not a.content or a.content.strip() == ''])
            })
            
        except Exception as e:
            orchestrator.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error_message=str(e)
            )
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/task/<task_id>/rerun', methods=['POST'])
    def rerun_task(task_id):
        """Re-run a specific task."""
        def execute(task):
            # Simplified re-execution
            task.status = TaskStatus.COMPLETED
            return task
        
        result = monitoring_agent.rerun_task(task_id, execute)
        
        if result:
            return jsonify({
                "success": True,
                "task": result.to_dict()
            })
        return jsonify({
            "success": False,
            "error": "Task not found or execution failed"
        }), 404
    
    # -------------- Artifact Routes --------------
    
    @app.route('/api/artifacts')
    def get_artifacts():
        """Get all generated artifacts."""
        if not current_pipeline["state"]:
            return jsonify({"artifacts": []})
        
        return jsonify({
            "artifacts": [a.to_dict() for a in current_pipeline["state"].artifacts]
        })
    
    @app.route('/api/artifact/<artifact_id>')
    def get_artifact(artifact_id):
        """Get a specific artifact."""
        if not current_pipeline["state"]:
            return jsonify({"error": "No pipeline"}), 404
        
        artifact = next(
            (a for a in current_pipeline["state"].artifacts if a.id == artifact_id),
            None
        )
        
        if artifact:
            return jsonify(artifact.to_dict())
        return jsonify({"error": "Artifact not found"}), 404
    
    @app.route('/api/artifact/<artifact_id>/diff')
    def get_artifact_diff(artifact_id):
        """Get diff for an artifact."""
        diff = monitoring_agent.get_artifact_diff(artifact_id)
        if diff:
            return jsonify(diff)
        return jsonify({"error": "No diff available"}), 404
    
    @app.route('/api/artifact/<artifact_id>/regenerate', methods=['POST'])
    def regenerate_artifact(artifact_id):
        """Regenerate a specific artifact."""
        def regenerate():
            # Simplified regeneration
            if not current_pipeline["state"]:
                raise Exception("No pipeline")
            
            artifact = next(
                (a for a in current_pipeline["state"].artifacts if a.id == artifact_id),
                None
            )
            if not artifact:
                raise Exception("Artifact not found")
            
            # In real implementation, would re-execute the generating agent
            return artifact
        
        result = monitoring_agent.regenerate_artifact(artifact_id, regenerate)
        
        if result:
            return jsonify({
                "success": True,
                "artifact": result.to_dict()
            })
        return jsonify({
            "success": False,
            "error": "Regeneration failed"
        }), 500
    
    @app.route('/api/artifacts/export', methods=['GET'])
    def export_all_artifacts():
        """Export all artifacts as a ZIP file."""
        if not current_pipeline["state"]:
            return jsonify({"error": "No pipeline with artifacts to export"}), 404
        
        artifacts = current_pipeline["state"].artifacts
        
        if not artifacts:
            return jsonify({"error": "No artifacts generated yet"}), 404
        
        # Create a ZIP file in memory
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all code artifacts
            for artifact in artifacts:
                if artifact.content and artifact.content.strip():
                    zf.writestr(artifact.file_path, artifact.content)
            
            # Add a manifest file with metadata about all artifacts
            manifest = {
                "generated_at": datetime.now().isoformat(),
                "total_artifacts": len(artifacts),
                "artifacts": []
            }
            
            for artifact in artifacts:
                if artifact.content and artifact.content.strip():
                    artifact_info = {
                        "file_path": artifact.file_path,
                        "artifact_type": artifact.artifact_type if hasattr(artifact, 'artifact_type') else "unknown",
                        "language": artifact.language if hasattr(artifact, 'language') else "unknown",
                        "size_bytes": len(artifact.content),
                        "documentation": artifact.documentation if hasattr(artifact, 'documentation') and artifact.documentation else None
                    }
                    manifest["artifacts"].append(artifact_info)
            
            # Add manifest.json
            zf.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
            
            # Add README with instructions
            readme_content = f"""# Generated Code Package

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total files: {len([a for a in artifacts if a.content and a.content.strip()])}

## Contents

This package contains all the generated code artifacts from the Agentic Code Generation System.

### Files Generated:
"""
            for artifact in artifacts:
                if artifact.content and artifact.content.strip():
                    readme_content += f"\n- {artifact.file_path}"
                    if hasattr(artifact, 'documentation') and artifact.documentation:
                        readme_content += f"\n  {artifact.documentation[:100]}..."
            
            readme_content += """

## Usage

1. Extract this ZIP file to your desired location
2. Review the MANIFEST.json file for details about each artifact
3. Integrate the generated code into your project
4. Run tests to ensure everything works correctly

## Support

For more information about the Agentic Code Generation System, 
please refer to the project documentation.
"""
            zf.writestr('README.md', readme_content)
        
        memory_file.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'generated_code_{timestamp}.zip'
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
    
    @app.route('/api/artifacts/export-with-docs', methods=['GET'])
    def export_all_artifacts_with_docs():
        """Export all artifacts with comprehensive documentation."""
        if not current_pipeline["state"]:
            return jsonify({"error": "No pipeline with artifacts to export"}), 404
        
        artifacts = current_pipeline["state"].artifacts
        
        if not artifacts:
            return jsonify({"error": "No artifacts generated yet"}), 404
        
        # Create a ZIP file in memory with enhanced documentation
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all code artifacts in a 'src' directory
            for artifact in artifacts:
                if artifact.content and artifact.content.strip():
                    zf.writestr(f"src/{artifact.file_path}", artifact.content)
            
            # Create comprehensive documentation
            docs_dir = "docs/"
            
            # Create artifact documentation for each file
            for artifact in artifacts:
                if artifact.content and artifact.content.strip():
                    doc_filename = artifact.file_path.replace('/', '_').replace('.', '_') + '.md'
                    doc_content = f"""# {artifact.file_path}

## Metadata
- **Type**: {artifact.artifact_type if hasattr(artifact, 'artifact_type') else 'N/A'}
- **Language**: {artifact.language if hasattr(artifact, 'language') else 'N/A'}
- **Size**: {len(artifact.content)} bytes
- **Generated**: {artifact.created_at.isoformat() if hasattr(artifact, 'created_at') else 'N/A'}

## Documentation
{artifact.documentation if hasattr(artifact, 'documentation') and artifact.documentation else 'No documentation available.'}

## Code Preview
```
{artifact.content[:500]}
{'...' if len(artifact.content) > 500 else ''}
```
"""
                    zf.writestr(f"{docs_dir}{doc_filename}", doc_content)
            
            # Create main documentation index
            index_content = f"""# Generated Code Documentation

Package generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total artifacts: {len([a for a in artifacts if a.content and a.content.strip()])}

## Structure

- `src/` - Contains all generated source code files
- `docs/` - Contains documentation for each artifact
- `MANIFEST.json` - Metadata about all artifacts
- `README.md` - Quick start guide

## Artifacts

"""
            for artifact in artifacts:
                if artifact.content and artifact.content.strip():
                    index_content += f"### {artifact.file_path}\n"
                    index_content += f"- Type: {artifact.artifact_type if hasattr(artifact, 'artifact_type') else 'N/A'}\n"
                    index_content += f"- Language: {artifact.language if hasattr(artifact, 'language') else 'N/A'}\n"
                    if hasattr(artifact, 'documentation') and artifact.documentation:
                        index_content += f"- {artifact.documentation[:150]}...\n"
                    index_content += "\n"
            
            zf.writestr(f"{docs_dir}INDEX.md", index_content)
            
            # Add manifest
            manifest = {
                "generated_at": datetime.now().isoformat(),
                "export_type": "with_documentation",
                "total_artifacts": len([a for a in artifacts if a.content and a.content.strip()]),
                "artifacts": [
                    {
                        "file_path": a.file_path,
                        "artifact_type": a.artifact_type if hasattr(a, 'artifact_type') else "unknown",
                        "language": a.language if hasattr(a, 'language') else "unknown",
                        "size_bytes": len(a.content),
                        "has_documentation": bool(hasattr(a, 'documentation') and a.documentation)
                    }
                    for a in artifacts if a.content and a.content.strip()
                ]
            }
            zf.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
            
            # Add README
            readme = """# Generated Code Package (With Documentation)

This package contains all generated code along with comprehensive documentation.

## Directory Structure

```
├── src/              # All source code files
├── docs/             # Documentation for each artifact
│   └── INDEX.md      # Documentation index
├── MANIFEST.json     # Metadata and manifest
└── README.md         # This file
```

## Quick Start

1. Extract this ZIP file
2. Review `docs/INDEX.md` for an overview
3. Check individual file documentation in `docs/`
4. Source code is in the `src/` directory

## Integration

To integrate this code into your project:
1. Copy files from `src/` to your project structure
2. Review dependencies in MANIFEST.json
3. Run your build system
4. Execute tests

## Documentation

Each generated file has corresponding documentation in the `docs/` directory
that explains its purpose, structure, and usage.
"""
            zf.writestr('README.md', readme)
        
        memory_file.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'generated_code_with_docs_{timestamp}.zip'
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
    
    @app.route('/api/artifact/<artifact_id>/export', methods=['GET'])
    def export_single_artifact(artifact_id):
        """Export a single artifact as a downloadable file."""
        if not current_pipeline["state"]:
            return jsonify({"error": "No pipeline"}), 404
        
        artifact = next(
            (a for a in current_pipeline["state"].artifacts if a.id == artifact_id),
            None
        )
        
        if not artifact:
            return jsonify({"error": "Artifact not found"}), 404
        
        if not artifact.content or not artifact.content.strip():
            return jsonify({"error": "Artifact has no content"}), 404
        
        # Create file in memory
        memory_file = io.BytesIO()
        memory_file.write(artifact.content.encode('utf-8'))
        memory_file.seek(0)
        
        # Get just the filename from the path
        filename = artifact.file_path.split('/')[-1]
        
        # Determine mimetype based on extension
        mimetype = 'text/plain'
        if filename.endswith('.py'):
            mimetype = 'text/x-python'
        elif filename.endswith(('.js', '.jsx')):
            mimetype = 'text/javascript'
        elif filename.endswith(('.ts', '.tsx')):
            mimetype = 'text/typescript'
        elif filename.endswith('.html'):
            mimetype = 'text/html'
        elif filename.endswith('.css'):
            mimetype = 'text/css'
        elif filename.endswith('.json'):
            mimetype = 'application/json'
        
        return send_file(
            memory_file,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    
    @app.route('/api/pipeline/export-complete', methods=['GET'])
    def export_complete_pipeline():
        """Export the complete pipeline output including all results, artifacts, tests, and reports."""
        if not current_pipeline["state"]:
            return jsonify({"error": "No pipeline to export"}), 404
        
        pipeline = current_pipeline["state"]
        
        # Create a comprehensive ZIP package
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 1. Add all generated code artifacts
            code_artifacts = [a for a in pipeline.artifacts if a.content and a.content.strip()]
            for artifact in code_artifacts:
                zf.writestr(f"code/{artifact.file_path}", artifact.content)
            
            # 2. Add pipeline execution report
            pipeline_report = {
                "pipeline_id": pipeline.id,
                "status": pipeline.status.value,
                "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
                "updated_at": pipeline.updated_at.isoformat() if pipeline.updated_at else None,
                "summary": {
                    "total_tasks": len(pipeline.tasks),
                    "completed_tasks": sum(1 for t in pipeline.tasks if t.status.value == 'completed'),
                    "failed_tasks": sum(1 for t in pipeline.tasks if t.status.value == 'failed'),
                    "total_artifacts": len(code_artifacts),
                    "total_test_reports": len(pipeline.test_reports),
                    "total_test_plans": len(pipeline.test_plans)
                },
                "tasks": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "status": t.status.value,
                        "agent_type": t.agent_type.value,
                        "error_message": t.error_message
                    }
                    for t in pipeline.tasks
                ]
            }
            zf.writestr('reports/pipeline_execution.json', json.dumps(pipeline_report, indent=2))
            
            # 3. Add test reports
            if pipeline.test_reports:
                for i, test_report in enumerate(pipeline.test_reports):
                    report_data = test_report.to_dict()
                    zf.writestr(f'reports/test_report_{i+1}.json', json.dumps(report_data, indent=2))
                
                # Add summary of all tests
                test_summary = {
                    "total_test_reports": len(pipeline.test_reports),
                    "overall_stats": {
                        "total_tests": sum(r.total_tests for r in pipeline.test_reports),
                        "passed_tests": sum(r.passed_tests for r in pipeline.test_reports),
                        "failed_tests": sum(r.failed_tests for r in pipeline.test_reports),
                        "average_coverage": sum(r.overall_coverage for r in pipeline.test_reports) / len(pipeline.test_reports) if pipeline.test_reports else 0
                    }
                }
                zf.writestr('reports/test_summary.json', json.dumps(test_summary, indent=2))
            
            # 4. Add test plans
            if pipeline.test_plans:
                for i, test_plan in enumerate(pipeline.test_plans):
                    plan_data = test_plan.to_dict()
                    zf.writestr(f'test_plans/test_plan_{i+1}.json', json.dumps(plan_data, indent=2))
            
            # 5. Add original specification
            if pipeline.spec:
                spec_data = pipeline.spec.to_dict()
                zf.writestr('specification/requirements.json', json.dumps(spec_data, indent=2))
                
                # Create readable version
                spec_readable = f"""# Project Specification

## User Stories ({len(pipeline.spec.user_stories)})

"""
                for story in pipeline.spec.user_stories:
                    spec_readable += f"""### {story.id}: {story.title}

**Description:** {story.description}

**Acceptance Criteria:**
"""
                    for criterion in story.acceptance_criteria:
                        spec_readable += f"- {criterion}\n"
                    spec_readable += f"\n**Priority:** {story.priority}\n\n"
                
                if pipeline.spec.tech_stack:
                    spec_readable += f"\n## Technology Stack\n\n"
                    for key, value in pipeline.spec.tech_stack.items():
                        spec_readable += f"- **{key}:** {value}\n"
                
                zf.writestr('specification/requirements.md', spec_readable)
            
            # 6. Add pipeline logs
            if pipeline.logs:
                logs_content = ""
                for log_entry in pipeline.logs:
                    timestamp = log_entry.get('timestamp', 'N/A')
                    level = log_entry.get('level', 'INFO')
                    message = log_entry.get('message', '')
                    logs_content += f"[{timestamp}] [{level}] {message}\n"
                zf.writestr('logs/pipeline.log', logs_content)
            
            # 7. Create comprehensive manifest
            manifest = {
                "export_type": "complete_pipeline_output",
                "generated_at": datetime.now().isoformat(),
                "pipeline": {
                    "id": pipeline.id,
                    "status": pipeline.status.value,
                    "duration": None  # Could calculate if we have timestamps
                },
                "contents": {
                    "code_artifacts": len(code_artifacts),
                    "test_reports": len(pipeline.test_reports),
                    "test_plans": len(pipeline.test_plans),
                    "tasks": len(pipeline.tasks),
                    "has_specification": pipeline.spec is not None,
                    "has_logs": bool(pipeline.logs)
                },
                "structure": {
                    "code/": "All generated source code files",
                    "reports/": "Pipeline execution and test reports",
                    "test_plans/": "Generated test plans",
                    "specification/": "Original requirements and spec",
                    "logs/": "Pipeline execution logs",
                    "MANIFEST.json": "This file",
                    "README.md": "Quick start guide"
                }
            }
            zf.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
            
            # 8. Create comprehensive README
            readme = f"""# Complete Pipeline Output Package

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pipeline ID: {pipeline.id}
Status: {pipeline.status.value}

## Package Contents

This package contains the complete output from the Agentic Code Generation pipeline execution.

### Directory Structure

```
├── code/                    # All generated source code ({len(code_artifacts)} files)
├── reports/                 # Pipeline and test execution reports
│   ├── pipeline_execution.json
│   ├── test_report_*.json
│   └── test_summary.json
├── test_plans/              # Generated test plans
├── specification/           # Original requirements
│   ├── requirements.json    # Machine-readable spec
│   └── requirements.md      # Human-readable spec
├── logs/                    # Pipeline execution logs
├── MANIFEST.json            # Package metadata
└── README.md                # This file
```

## Pipeline Summary

- **Total Tasks:** {len(pipeline.tasks)}
- **Completed Tasks:** {sum(1 for t in pipeline.tasks if t.status.value == 'completed')}
- **Generated Artifacts:** {len(code_artifacts)}
- **Test Reports:** {len(pipeline.test_reports)}
"""
            
            if pipeline.test_reports:
                total_tests = sum(r.total_tests for r in pipeline.test_reports)
                passed_tests = sum(r.passed_tests for r in pipeline.test_reports)
                readme += f"""
## Test Results

- **Total Tests:** {total_tests}
- **Passed:** {passed_tests}
- **Failed:** {total_tests - passed_tests}
- **Pass Rate:** {(passed_tests / total_tests * 100) if total_tests > 0 else 0:.1f}%
"""
            
            readme += """

## Quick Start

1. **Review the Code:** Check the `code/` directory for all generated files
2. **Check Test Reports:** Review `reports/` for testing results
3. **Read Specification:** See `specification/` for requirements
4. **Review Execution:** Check `reports/pipeline_execution.json` for task details

## Integration

To integrate this code into your project:

1. Extract the ZIP file
2. Copy files from `code/` to your project structure
3. Review test reports to understand coverage
4. Check specification for requirements traceability
5. Review logs for any warnings or issues

## Support

This package was generated by the Agentic Code Generation System.
For questions or issues, refer to the pipeline execution report.
"""
            zf.writestr('README.md', readme)
        
        memory_file.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'pipeline_complete_output_{timestamp}.zip'
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
    
    # -------------- Test Routes --------------
    
    @app.route('/api/tests/run', methods=['POST'])
    def run_tests():
        """
        Run tests.
        
        Request body:
        {
            "test_path": "tests/",
            "test_type": "unit" | "integration" | "e2e"
        }
        """
        try:
            body = request.get_json()
            test_path = body.get('test_path', 'tests/')
            test_type = body.get('test_type', 'unit')
            
            from ..models import TestType
            type_map = {
                'unit': TestType.UNIT,
                'integration': TestType.INTEGRATION,
                'e2e': TestType.E2E
            }
            
            report = testing_agent.run_tests(
                test_path, 
                type_map.get(test_type, TestType.UNIT)
            )
            
            # Add report to pipeline
            orchestrator.add_test_report(report)
            
            return jsonify({
                "success": True,
                "report": report.to_dict(),
                "summary": testing_agent.generate_report(report)
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/tests/reports')
    def get_test_reports():
        """Get all test reports."""
        if not current_pipeline["state"]:
            return jsonify({"reports": []})
        
        return jsonify({
            "reports": [r.to_dict() for r in current_pipeline["state"].test_reports]
        })
    
    @app.route('/api/tests/summary')
    def get_test_summary():
        """Get test results summary."""
        return jsonify(monitoring_agent.get_test_results_summary())
    
    # -------------- Legacy Analysis Routes --------------
    
    @app.route('/api/legacy/analyze', methods=['POST'])
    def analyze_legacy():
        """
        Analyze a legacy repository.
        
        Request body:
        {
            "repo_path": "/path/to/repo"
        }
        """
        try:
            body = request.get_json()
            repo_path = body.get('repo_path', '')
            
            analysis = legacy_agent.analyze_repository(repo_path)
            
            return jsonify({
                "success": True,
                "analysis": analysis.to_dict()
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/legacy/migration-plan', methods=['POST'])
    def get_migration_plan():
        """
        Generate migration plan.
        
        Request body:
        {
            "repo_path": "/path/to/repo",
            "target_stack": {...}
        }
        """
        try:
            body = request.get_json()
            repo_path = body.get('repo_path', '')
            target_stack = body.get('target_stack', {})
            
            analysis = legacy_agent.analyze_repository(repo_path)
            plan = legacy_agent.generate_migration_plan(analysis, target_stack)
            
            return jsonify({
                "success": True,
                "plan": plan
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # -------------- Prompt Refinement Routes --------------
    
    @app.route('/api/prompt/analyze', methods=['POST'])
    def analyze_prompt():
        """
        Analyze a prompt for issues.
        
        Request body:
        {
            "prompt": "...",
            "acceptance_criteria": [...] (optional)
        }
        """
        try:
            body = request.get_json()
            prompt = body.get('prompt', '')
            criteria = body.get('acceptance_criteria')
            
            issues = prompt_agent.analyze_prompt(prompt, criteria)
            
            return jsonify({
                "success": True,
                "issues": [
                    {
                        "type": i.issue_type.value,
                        "description": i.description,
                        "severity": i.severity,
                        "suggestion": i.suggestion
                    }
                    for i in issues
                ]
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @app.route('/api/prompt/refine', methods=['POST'])
    def refine_prompt():
        """
        Refine a prompt.
        
        Request body:
        {
            "prompt": "...",
            "acceptance_criteria": [...] (optional),
            "context": {...} (optional)
        }
        """
        try:
            body = request.get_json()
            prompt = body.get('prompt', '')
            criteria = body.get('acceptance_criteria')
            context = body.get('context')
            
            result = prompt_agent.refine_prompt(prompt, criteria, context)
            
            return jsonify({
                "success": True,
                "original": result.original_prompt,
                "refined": result.refined_prompt,
                "improvements": result.improvements,
                "confidence_score": result.confidence_score
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # -------------- Monitoring Routes --------------
    
    @app.route('/api/monitoring/dashboard')
    def get_dashboard():
        """Get dashboard data."""
        return jsonify(monitoring_agent.get_dashboard_data())
    
    @app.route('/api/monitoring/events')
    def get_events():
        """Get recent events."""
        limit = request.args.get('limit', 100, type=int)
        events = monitoring_agent.get_events(limit=limit)
        return jsonify({
            "events": [e.to_dict() for e in events]
        })
    
    @app.route('/api/monitoring/logs')
    def get_logs():
        """Get pipeline logs."""
        level = request.args.get('level')
        limit = request.args.get('limit', 100, type=int)
        logs = monitoring_agent.get_pipeline_logs(level=level, limit=limit)
        return jsonify({"logs": logs})
    
    @app.route('/api/monitoring/checkpoints')
    def get_checkpoints():
        """Get all checkpoints."""
        return jsonify({
            "checkpoints": [c.to_dict() for c in monitoring_agent.checkpoints]
        })
    
    @app.route('/api/monitoring/checkpoint', methods=['POST'])
    def create_checkpoint():
        """
        Create a checkpoint.
        
        Request body:
        {
            "name": "checkpoint_name"
        }
        """
        if not current_pipeline["state"]:
            return jsonify({"success": False, "error": "No pipeline"}), 404
        
        body = request.get_json()
        name = body.get('name', f'checkpoint_{datetime.now().isoformat()}')
        
        checkpoint = monitoring_agent.create_checkpoint(name, current_pipeline["state"])
        
        return jsonify({
            "success": True,
            "checkpoint": checkpoint.to_dict()
        })
    
    @app.route('/api/monitoring/rollback/<checkpoint_id>', methods=['POST'])
    def rollback(checkpoint_id):
        """Rollback to a checkpoint."""
        result = monitoring_agent.rollback_to_checkpoint(checkpoint_id)
        
        if result:
            return jsonify({
                "success": True,
                "restored_state": result
            })
        return jsonify({
            "success": False,
            "error": "Checkpoint not found"
        }), 404
    
    # -------------- Automation & Configuration Routes --------------
    
    @app.route('/api/config/auto-mode', methods=['POST'])
    def configure_auto_mode():
        """
        Enable or disable automatic mode.
        
        Request body:
        {
            "enabled": true,
            "ado_config": {
                "org_url": "...",
                "pat": "...",
                "project": "...",
                "repo_name": "...",
                "branch": "refs/heads/generated-code" (optional)
            }
        }
        """
        try:
            body = request.get_json(force=True, silent=False)
            enabled = body.get('enabled', True)
            ado_config = body.get('ado_config', {})
            
            # Store in global state (in production, use proper state management)
            app.config['AUTO_MODE'] = enabled
            app.config['ADO_CONFIG'] = ado_config
            
            response = jsonify({
                "success": True,
                "auto_mode": enabled,
                "ado_configured": bool(ado_config.get('org_url') and ado_config.get('pat'))
            })
            response.headers['Content-Type'] = 'application/json'
            return response
            
        except Exception as e:
            app.logger.error(f"[Config] Error: {e}")
            response = jsonify({
                "success": False,
                "error": str(e)
            })
            response.headers['Content-Type'] = 'application/json'
            return response, 400
    
    @app.route('/api/config/auto-mode', methods=['GET'])
    def get_auto_mode():
        """Get current auto mode configuration."""
        return jsonify({
            "auto_mode": app.config.get('AUTO_MODE', True),
            "ado_configured": bool(
                app.config.get('ADO_CONFIG', {}).get('org_url') and
                app.config.get('ADO_CONFIG', {}).get('pat')
            )
        })
    
    @app.route('/api/pipeline/execute-auto', methods=['POST'])
    def execute_pipeline_auto():
        """
        Execute the complete automated pipeline including:
        1. Optional legacy analysis
        2. Prompt refinement
        3. Code generation
        
        Request body:
        {
            "spec": {...},  // CanonicalSpec
            "legacy_repo_path": "..." (optional)
        }
        """
        try:
            body = request.get_json(force=True, silent=False)
            
            if not body:
                response = jsonify({
                    "success": False,
                    "error": "No JSON body provided"
                })
                response.headers['Content-Type'] = 'application/json'
                return response, 400
            
            spec_data = body.get('spec')
            legacy_repo_path = body.get('legacy_repo_path')
            
            if not spec_data:
                response = jsonify({
                    "success": False,
                    "error": "spec is required"
                })
                response.headers['Content-Type'] = 'application/json'
                return response, 400
            
            # Import and create crew with auto mode (no ADO config needed)
            from ..crew import CodeGenerationCrew
            crew = CodeGenerationCrew(auto_mode=True, ado_config=None)
            
            app.logger.info("[AutoPipeline] Starting automated pipeline execution")
            
            # Reconstruct spec
            stories = []
            for s in spec_data.get('user_stories', []):
                try:
                    from ..models import UserStory
                    story = UserStory(
                        id=s.get('id', ''),
                        title=s.get('title', ''),
                        description=s.get('description', ''),
                        acceptance_criteria=s.get('acceptance_criteria', []),
                        persona=s.get('persona'),
                        priority=s.get('priority', 3),
                        non_functional_hints=s.get('non_functional_hints', []),
                        tags=s.get('tags', [])
                    )
                    stories.append(story)
                except Exception as e:
                    app.logger.warning(f"[AutoPipeline] Skipping malformed story: {e}")
                    continue
            
            from ..models import CanonicalSpec
            spec = CanonicalSpec(
                user_stories=stories,
                requirements=spec_data.get('requirements', {}),
                tech_stack=spec_data.get('tech_stack', {}),
                constraints=spec_data.get('constraints', {}),
                project_name=spec_data.get('project_name', 'Generated Code')
            )
            
            # Step 1: Analyze legacy if path provided
            if legacy_repo_path:
                app.logger.info(f"[AutoPipeline] Analyzing legacy repository: {legacy_repo_path}")
                try:
                    legacy_analysis = crew.analyze_legacy(legacy_repo_path)
                    app.logger.info(f"[AutoPipeline] Legacy analysis complete")
                except Exception as e:
                    app.logger.warning(f"[AutoPipeline] Legacy analysis failed: {e}")
            
            # Step 2: Build pipeline
            app.logger.info("[AutoPipeline] Building pipeline")
            pipeline = crew.build_pipeline(spec)
            current_pipeline["state"] = pipeline
            monitoring_agent.set_pipeline(pipeline)
            
            # Step 3: Execute pipeline (without auto-commit)
            app.logger.info("[AutoPipeline] Executing pipeline")
            result_pipeline = crew.execute_pipeline(parallel=True, auto_commit=False)
            
            app.logger.info("[AutoPipeline] Pipeline execution complete")
            
            response_data = {
                "success": True,
                "pipeline": result_pipeline.to_dict(),
                "artifacts_count": len(result_pipeline.artifacts),
                "completed_tasks": sum(1 for t in result_pipeline.tasks if t.status.value == 'completed'),
                "total_tasks": len(result_pipeline.tasks)
            }
            
            response = jsonify(response_data)
            response.headers['Content-Type'] = 'application/json'
            return response
            
        except Exception as e:
            app.logger.error(f"[AutoPipeline] Failed: {str(e)}", exc_info=True)
            response = jsonify({
                "success": False,
                "error": str(e)
            })
            response.headers['Content-Type'] = 'application/json'
            return response, 500
    
    # ============== WebSocket Events ==============
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        emit('connected', {'status': 'connected'})
        # Send current state
        if current_pipeline["state"]:
            emit('pipeline_state', current_pipeline["state"].to_dict())
    
    @socketio.on('subscribe_events')
    def handle_subscribe():
        """Subscribe to event stream."""
        emit('subscribed', {'status': 'subscribed to events'})
    
    @socketio.on('get_status')
    def handle_get_status():
        """Get current status via WebSocket."""
        emit('status', {
            'task_summary': monitoring_agent.get_task_status_summary(),
            'test_summary': monitoring_agent.get_test_results_summary()
        })
    
    return app, socketio


# Create app instance
app, socketio = create_app()


def run_server(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask server with SocketIO."""
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
