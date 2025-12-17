"""
Monitoring & UI Agent
Streams event logs, task states, artifacts, test results, and diffs.
Enables safe re-runs, rollbacks, or regeneration of specific artifacts without restarting the whole pipeline.
"""

import difflib
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import os
import threading
from queue import Queue

from crewai import Agent

from ..utils.llm_config import get_llm

from ..models import (
    PipelineState, Task, TaskStatus, GeneratedArtifact,
    TestReport, AgentType
)


class EventType(Enum):
    """Types of events in the system."""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    ARTIFACT_CREATED = "artifact_created"
    TEST_COMPLETED = "test_completed"
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    LOG_MESSAGE = "log_message"


@dataclass
class Event:
    """Represents a system event."""
    id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    task_id: Optional[str] = None
    artifact_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "task_id": self.task_id,
            "artifact_id": self.artifact_id
        }


@dataclass
class Checkpoint:
    """Represents a pipeline checkpoint for rollback."""
    id: str
    name: str
    timestamp: datetime
    pipeline_state: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_state": self.pipeline_state,
            "artifacts_count": len(self.artifacts)
        }


class MonitoringAgent:

    def __init__(self, max_events: int = 1000, max_checkpoints: int = 10):
        """
        Initialize the Monitoring Agent.
        
        Args:
            max_events: Maximum events to keep in memory
            max_checkpoints: Maximum checkpoints to keep
        """
        self.agent_type = AgentType.MONITORING
        self.max_events = max_events
        self.max_checkpoints = max_checkpoints
        
        # Event storage and streaming
        self.events: List[Event] = []
        self.event_queue: Queue = Queue()
        self.subscribers: List[Callable[[Event], None]] = []
        
        # Checkpoint storage
        self.checkpoints: List[Checkpoint] = []
        
        # Artifact storage (for rollback)
        self.artifact_history: Dict[str, List[GeneratedArtifact]] = {}
        
        # Current pipeline state
        self.current_pipeline: Optional[PipelineState] = None
        
        # Event counter for IDs
        self._event_counter = 0
        self._lock = threading.Lock()
        
        # Initialize CrewAI agent
        self.llm = get_llm(temperature=0.1)
        
        self.crew_agent = Agent(
            role="DevOps Monitor",
            goal="Monitor pipeline execution and provide status updates",
            backstory="""You are an expert DevOps engineer who monitors CI/CD
            pipelines and provides clear status reports and recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        ) if self.llm else None
    
    def emit_event(self, event_type: EventType, data: Dict[str, Any],
                   task_id: Optional[str] = None,
                   artifact_id: Optional[str] = None) -> Event:
        """
        Emit a new event.
        
        Args:
            event_type: Type of event
            data: Event data
            task_id: Optional associated task ID
            artifact_id: Optional associated artifact ID
            
        Returns:
            Created event
        """
        with self._lock:
            self._event_counter += 1
            event_id = f"evt_{self._event_counter}"
        
        event = Event(
            id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            task_id=task_id,
            artifact_id=artifact_id
        )
        
        # Store event
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Add to queue for streaming
        self.event_queue.put(event)
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception:
                pass  # Don't let subscriber errors affect event emission
        
        return event
    
    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to events.
        
        Args:
            callback: Function to call for each event
        """
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """
        Unsubscribe from events.
        
        Args:
            callback: Function to remove
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_events(self, event_type: Optional[EventType] = None,
                   since: Optional[datetime] = None,
                   task_id: Optional[str] = None,
                   limit: int = 100) -> List[Event]:
        """
        Get filtered events.
        
        Args:
            event_type: Optional filter by event type
            since: Optional filter by timestamp
            task_id: Optional filter by task ID
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        if task_id:
            events = [e for e in events if e.task_id == task_id]
        
        return events[-limit:]
    
    def create_checkpoint(self, name: str, pipeline_state: PipelineState) -> Checkpoint:
        """
        Create a checkpoint for rollback.
        
        Args:
            name: Checkpoint name
            pipeline_state: Current pipeline state
            
        Returns:
            Created checkpoint
        """
        checkpoint = Checkpoint(
            id=f"chk_{len(self.checkpoints) + 1}",
            name=name,
            timestamp=datetime.now(),
            pipeline_state=pipeline_state.to_dict(),
            artifacts=[a.to_dict() for a in pipeline_state.artifacts]
        )
        
        self.checkpoints.append(checkpoint)
        
        # Keep only recent checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
        
        self.emit_event(EventType.LOG_MESSAGE, {
            "level": "info",
            "message": f"Checkpoint created: {name}"
        })
        
        return checkpoint
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Rollback to a previous checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            Restored pipeline state or None if not found
        """
        checkpoint = next(
            (c for c in self.checkpoints if c.id == checkpoint_id),
            None
        )
        
        if not checkpoint:
            return None
        
        self.emit_event(EventType.ROLLBACK_STARTED, {
            "checkpoint_id": checkpoint_id,
            "checkpoint_name": checkpoint.name
        })
        
        # In a real implementation, this would restore actual state
        self.emit_event(EventType.ROLLBACK_COMPLETED, {
            "checkpoint_id": checkpoint_id,
            "restored_at": datetime.now().isoformat()
        })
        
        return checkpoint.pipeline_state
    
    def regenerate_artifact(self, artifact_id: str,
                           regenerate_func: Callable[[], GeneratedArtifact]) -> Optional[GeneratedArtifact]:
        """
        Regenerate a specific artifact without restarting pipeline.
        
        Args:
            artifact_id: ID of artifact to regenerate
            regenerate_func: Function to call to regenerate artifact
            
        Returns:
            New artifact or None if failed
        """
        self.emit_event(EventType.LOG_MESSAGE, {
            "level": "info",
            "message": f"Regenerating artifact: {artifact_id}"
        })
        
        try:
            # Store old version in history
            if self.current_pipeline:
                old_artifact = next(
                    (a for a in self.current_pipeline.artifacts if a.id == artifact_id),
                    None
                )
                if old_artifact:
                    if artifact_id not in self.artifact_history:
                        self.artifact_history[artifact_id] = []
                    self.artifact_history[artifact_id].append(old_artifact)
            
            # Regenerate
            new_artifact = regenerate_func()
            
            self.emit_event(EventType.ARTIFACT_CREATED, {
                "artifact_id": new_artifact.id,
                "file_path": new_artifact.file_path,
                "regenerated": True
            }, artifact_id=new_artifact.id)
            
            return new_artifact
            
        except Exception as e:
            self.emit_event(EventType.LOG_MESSAGE, {
                "level": "error",
                "message": f"Failed to regenerate artifact: {e}"
            })
            return None
    
    def rerun_task(self, task_id: str,
                   execute_func: Callable[[Task], Task]) -> Optional[Task]:
        """
        Re-run a specific task without restarting pipeline.
        
        Args:
            task_id: ID of task to re-run
            execute_func: Function to execute the task
            
        Returns:
            Updated task or None if failed
        """
        if not self.current_pipeline:
            return None
        
        task = next(
            (t for t in self.current_pipeline.tasks if t.id == task_id),
            None
        )
        
        if not task:
            return None
        
        self.emit_event(EventType.TASK_STARTED, {
            "task_id": task_id,
            "task_name": task.name,
            "rerun": True
        }, task_id=task_id)
        
        try:
            # Reset task status
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            task.error_message = None
            
            # Execute
            result = execute_func(task)
            
            self.emit_event(EventType.TASK_COMPLETED, {
                "task_id": task_id,
                "task_name": task.name,
                "rerun": True
            }, task_id=task_id)
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            self.emit_event(EventType.TASK_FAILED, {
                "task_id": task_id,
                "error": str(e)
            }, task_id=task_id)
            
            return task
    
    def get_task_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of all task statuses.
        
        Returns:
            Dictionary with status summary
        """
        if not self.current_pipeline:
            return {"error": "No pipeline active"}
        
        status_counts = {}
        for task in self.current_pipeline.tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total = len(self.current_pipeline.tasks)
        completed = status_counts.get(TaskStatus.COMPLETED.value, 0)
        
        return {
            "total_tasks": total,
            "status_counts": status_counts,
            "progress_percentage": (completed / total * 100) if total > 0 else 0,
            "pipeline_status": self.current_pipeline.status.value
        }
    
    def get_test_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of all test results.
        
        Returns:
            Dictionary with test summary
        """
        if not self.current_pipeline:
            return {"error": "No pipeline active"}
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        overall_coverage = 0.0
        
        for report in self.current_pipeline.test_reports:
            total_tests += report.total_tests
            passed_tests += report.passed_tests
            failed_tests += report.failed_tests
            overall_coverage += report.overall_coverage
        
        report_count = len(self.current_pipeline.test_reports)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "average_coverage": (overall_coverage / report_count) if report_count > 0 else 0,
            "report_count": report_count
        }
    
    def get_artifact_diff(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        Get diff between current and previous artifact versions.
        
        Args:
            artifact_id: Artifact ID
            
        Returns:
            Diff information or None if no history
        """
        if artifact_id not in self.artifact_history:
            return None
        
        history = self.artifact_history[artifact_id]
        if not history:
            return None
        
        previous = history[-1]
        
        # Get current
        current = None
        if self.current_pipeline:
            current = next(
                (a for a in self.current_pipeline.artifacts if a.id == artifact_id),
                None
            )
        
        if not current:
            return None
        
        # Simple line-by-line diff
        prev_lines = previous.content.splitlines()
        curr_lines = current.content.splitlines()
        
        diff = {
            "artifact_id": artifact_id,
            "file_path": current.file_path,
            "previous_version": len(history),
            "current_version": len(history) + 1,
            "additions": 0,
            "deletions": 0,
            "changes": []
        }
        
        # Generate unified diff
        differ = difflib.unified_diff(
            prev_lines, curr_lines,
            fromfile=f"v{len(history)}",
            tofile=f"v{len(history) + 1}",
            lineterm=""
        )
        
        for line in differ:
            diff["changes"].append(line)
            if line.startswith('+') and not line.startswith('+++'):
                diff["additions"] += 1
            elif line.startswith('-') and not line.startswith('---'):
                diff["deletions"] += 1
        
        return diff
    
    def get_pipeline_logs(self, level: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get pipeline logs.
        
        Args:
            level: Optional filter by log level
            limit: Maximum logs to return
            
        Returns:
            List of log entries
        """
        log_events = [
            e for e in self.events 
            if e.event_type == EventType.LOG_MESSAGE
        ]
        
        if level:
            log_events = [
                e for e in log_events 
                if e.data.get("level") == level
            ]
        
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "level": e.data.get("level", "info"),
                "message": e.data.get("message", "")
            }
            for e in log_events[-limit:]
        ]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all data needed for the UI dashboard.
        
        Returns:
            Complete dashboard data
        """
        return {
            "task_summary": self.get_task_status_summary(),
            "test_summary": self.get_test_results_summary(),
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "recent_events": [e.to_dict() for e in self.events[-20:]],
            "recent_logs": self.get_pipeline_logs(limit=50)
        }
    
    def set_pipeline(self, pipeline: PipelineState) -> None:
        """
        Set the current pipeline to monitor.
        
        Args:
            pipeline: Pipeline state to monitor
        """
        self.current_pipeline = pipeline
        
        self.emit_event(EventType.PIPELINE_STARTED, {
            "pipeline_id": pipeline.id,
            "task_count": len(pipeline.tasks)
        })
    
    def get_event_stream_json(self) -> str:
        """
        Get events as JSON for streaming.
        
        Returns:
            JSON string of recent events
        """
        return json.dumps([e.to_dict() for e in self.events[-50:]])
