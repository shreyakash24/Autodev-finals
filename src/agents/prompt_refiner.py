"""
Prompt Refinement Engine
Analyzes prompts and responses for hallucinations, ambiguity, and unmet acceptance criteria.
Iteratively improves prompts to strengthen code quality, structure, and testability.
Feeds learnings back into Coordinator to update the build plan or adjust constraints.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os

from crewai import Agent

from ..utils.llm_config import get_llm

from ..models import AgentType, UserStory, CanonicalSpec


class IssueType(Enum):
    """Types of issues detected in prompts/responses."""
    HALLUCINATION = "hallucination"
    AMBIGUITY = "ambiguity"
    UNMET_CRITERIA = "unmet_criteria"
    MISSING_CONTEXT = "missing_context"
    POOR_STRUCTURE = "poor_structure"
    UNTESTABLE = "untestable"


@dataclass
class PromptIssue:
    """Represents an issue found in a prompt or response."""
    issue_type: IssueType
    description: str
    severity: str  # 'high', 'medium', 'low'
    suggestion: str
    location: Optional[str] = None


@dataclass
class RefinementResult:
    """Result of prompt refinement."""
    original_prompt: str
    refined_prompt: str
    issues_found: List[PromptIssue]
    improvements: List[str]
    confidence_score: float  # 0.0 to 1.0


class PromptRefinerAgent:
    """
    Agent that refines prompts and analyzes responses for quality.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Prompt Refinement Engine                              │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Prompt Analyzer                                │ │
    │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │ │
    │  │  │Hallucination│  │  Ambiguity   │  │ Criteria Checker          │  │ │
    │  │  │  Detector   │  │  Detector    │  │                           │  │ │
    │  │  └─────────────┘  └──────────────┘  └───────────────────────────┘  │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Prompt Optimizer                               │ │
    │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │ │
    │  │  │ Clarifier   │  │ Structurer   │  │ Testability Enhancer      │  │ │
    │  │  └─────────────┘  └──────────────┘  └───────────────────────────┘  │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    │                                                                          │
    │  ┌────────────────────────────────────────────────────────────────────┐ │
    │  │                      Learning Engine                                │ │
    │  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │ │
    │  │  │ Pattern DB  │  │ Feedback     │  │ Coordinator Feedback      │  │ │
    │  │  │             │  │ Aggregator   │  │                           │  │ │
    │  │  └─────────────┘  └──────────────┘  └───────────────────────────┘  │ │
    │  └────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    # Patterns that might indicate hallucination
    HALLUCINATION_PATTERNS = [
        r'definitely\s+will',
        r'guaranteed\s+to',
        r'always\s+works',
        r'never\s+fails',
        r'100%\s+accurate',
        r'impossib(?:le|ly)',
        r'cannot\s+fail',
    ]
    
    # Patterns that indicate ambiguity
    AMBIGUITY_PATTERNS = [
        r'\b(?:some|sometimes|maybe|might|could|possibly|perhaps)\b',
        r'\b(?:etc|and so on|and more|and others)\b',
        r'\b(?:appropriate|suitable|proper|relevant)\b(?!\s+\w+)',
        r'\b(?:various|several|multiple|many|few)\b(?!\s+\w+)',
        r'\b(?:usually|often|generally|typically)\b',
    ]
    
    # Keywords for testability
    TESTABILITY_KEYWORDS = [
        'assert', 'expect', 'verify', 'validate', 'check',
        'should', 'must', 'when', 'given', 'then',
        'input', 'output', 'result', 'return'
    ]
    
    def __init__(self):
        """Initialize the Prompt Refiner Agent."""
        self.agent_type = AgentType.PROMPT_REFINER
        self.learned_patterns: List[Dict[str, Any]] = []
        
        self.llm = get_llm(temperature=0.1)
        
        self.crew_agent = Agent(
            role="Prompt Engineer",
            goal="Refine prompts for optimal code generation quality",
            backstory="""You are an expert prompt engineer who analyzes and
            improves prompts to eliminate ambiguity, prevent hallucinations,
            and ensure generated code meets acceptance criteria.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        ) if self.llm else None
    
    def analyze_prompt(self, prompt: str, 
                      acceptance_criteria: Optional[List[str]] = None) -> List[PromptIssue]:
        """
        Analyze a prompt for potential issues.
        
        Args:
            prompt: The prompt to analyze
            acceptance_criteria: Optional list of criteria to check against
            
        Returns:
            List of issues found
        """
        issues = []
        
        # Check for hallucination indicators
        issues.extend(self._check_hallucination_patterns(prompt))
        
        # Check for ambiguity
        issues.extend(self._check_ambiguity(prompt))
        
        # Check against acceptance criteria
        if acceptance_criteria:
            issues.extend(self._check_criteria_coverage(prompt, acceptance_criteria))
        
        # Check structure
        issues.extend(self._check_structure(prompt))
        
        # Check testability
        issues.extend(self._check_testability(prompt))
        
        return issues
    
    def analyze_response(self, prompt: str, response: str,
                        acceptance_criteria: Optional[List[str]] = None) -> List[PromptIssue]:
        """
        Analyze a response for quality issues.
        
        Args:
            prompt: The original prompt
            response: The generated response
            acceptance_criteria: Optional criteria to verify
            
        Returns:
            List of issues found
        """
        issues = []
        
        # Check for hallucinations in response
        issues.extend(self._check_response_hallucinations(prompt, response))
        
        # Check if response addresses the prompt
        issues.extend(self._check_response_relevance(prompt, response))
        
        # Check criteria satisfaction
        if acceptance_criteria:
            issues.extend(self._check_criteria_met(response, acceptance_criteria))
        
        # Check code quality if response contains code
        if self._contains_code(response):
            issues.extend(self._check_code_quality(response))
        
        return issues
    
    def refine_prompt(self, prompt: str,
                     acceptance_criteria: Optional[List[str]] = None,
                     context: Optional[Dict[str, Any]] = None) -> RefinementResult:
        """
        Refine a prompt to improve quality.
        
        Args:
            prompt: Original prompt
            acceptance_criteria: Criteria to incorporate
            context: Additional context for refinement
            
        Returns:
            RefinementResult with refined prompt and details
        """
        issues = self.analyze_prompt(prompt, acceptance_criteria)
        improvements = []
        refined = prompt
        
        # Apply refinements based on issues
        for issue in issues:
            refinement = self._apply_refinement(refined, issue)
            if refinement != refined:
                refined = refinement
                improvements.append(f"Fixed {issue.issue_type.value}: {issue.description}")
        
        # Add structure if missing
        if not self._has_good_structure(prompt):
            refined = self._add_structure(refined, acceptance_criteria, context)
            improvements.append("Added structured format")
        
        # Add testability hints
        if not self._is_testable(prompt):
            refined = self._add_testability_hints(refined)
            improvements.append("Added testability hints")
        
        # Calculate confidence score
        remaining_issues = self.analyze_prompt(refined, acceptance_criteria)
        confidence = self._calculate_confidence(remaining_issues)
        
        return RefinementResult(
            original_prompt=prompt,
            refined_prompt=refined,
            issues_found=issues,
            improvements=improvements,
            confidence_score=confidence
        )
    
    def _check_hallucination_patterns(self, text: str) -> List[PromptIssue]:
        """Check for hallucination indicators."""
        issues = []
        text_lower = text.lower()
        
        for pattern in self.HALLUCINATION_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                issues.append(PromptIssue(
                    issue_type=IssueType.HALLUCINATION,
                    description=f"Potential overpromise: '{match.group()}'",
                    severity="medium",
                    suggestion="Use more measured language with realistic expectations",
                    location=f"Position {match.start()}-{match.end()}"
                ))
        
        return issues
    
    def _check_ambiguity(self, text: str) -> List[PromptIssue]:
        """Check for ambiguous language."""
        issues = []
        text_lower = text.lower()
        
        for pattern in self.AMBIGUITY_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                issues.append(PromptIssue(
                    issue_type=IssueType.AMBIGUITY,
                    description=f"Ambiguous term: '{match.group()}'",
                    severity="low",
                    suggestion="Be more specific about quantities, conditions, or options",
                    location=f"Position {match.start()}-{match.end()}"
                ))
        
        return issues
    
    def _check_criteria_coverage(self, prompt: str, 
                                criteria: List[str]) -> List[PromptIssue]:
        """Check if prompt covers all acceptance criteria."""
        issues = []
        prompt_lower = prompt.lower()
        
        for criterion in criteria:
            # Extract key terms from criterion
            key_terms = self._extract_key_terms(criterion)
            
            # Check if any key term is present
            found = any(term in prompt_lower for term in key_terms)
            
            if not found:
                issues.append(PromptIssue(
                    issue_type=IssueType.UNMET_CRITERIA,
                    description=f"Missing criterion: {criterion[:100]}",
                    severity="high",
                    suggestion=f"Add reference to: {', '.join(key_terms[:3])}"
                ))
        
        return issues
    
    def _check_structure(self, prompt: str) -> List[PromptIssue]:
        """Check prompt structure quality."""
        issues = []
        
        # Check length
        if len(prompt) < 50:
            issues.append(PromptIssue(
                issue_type=IssueType.POOR_STRUCTURE,
                description="Prompt is too short - may lack necessary context",
                severity="medium",
                suggestion="Add more context about requirements and constraints"
            ))
        
        # Check for sections
        has_sections = any(marker in prompt for marker in ['##', '**', ':', '-', '1.', '•'])
        if len(prompt) > 200 and not has_sections:
            issues.append(PromptIssue(
                issue_type=IssueType.POOR_STRUCTURE,
                description="Long prompt without clear sections",
                severity="low",
                suggestion="Break into sections with headers or bullet points"
            ))
        
        return issues
    
    def _check_testability(self, prompt: str) -> List[PromptIssue]:
        """Check if prompt leads to testable outputs."""
        issues = []
        prompt_lower = prompt.lower()
        
        # Check for testability keywords
        has_testability = any(kw in prompt_lower for kw in self.TESTABILITY_KEYWORDS)
        
        if not has_testability:
            issues.append(PromptIssue(
                issue_type=IssueType.UNTESTABLE,
                description="Prompt lacks testability hints",
                severity="medium",
                suggestion="Add expected inputs/outputs or verification criteria"
            ))
        
        return issues
    
    def _check_response_hallucinations(self, prompt: str, 
                                       response: str) -> List[PromptIssue]:
        """Check response for potential hallucinations."""
        issues = []
        
        # Check for claims not supported by prompt
        response_sentences = response.split('.')
        prompt_lower = prompt.lower()
        
        for sentence in response_sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Check for strong claims
            strong_claims = ['always', 'never', 'guaranteed', 'impossible', 'definitely']
            for claim in strong_claims:
                if claim in sentence.lower() and claim not in prompt_lower:
                    issues.append(PromptIssue(
                        issue_type=IssueType.HALLUCINATION,
                        description=f"Response contains unsupported claim: '{sentence[:100]}'",
                        severity="high",
                        suggestion="Verify claim or soften language"
                    ))
                    break
        
        return issues
    
    def _check_response_relevance(self, prompt: str, response: str) -> List[PromptIssue]:
        """Check if response is relevant to prompt."""
        issues = []
        
        # Extract key terms from prompt
        prompt_terms = set(self._extract_key_terms(prompt))
        response_terms = set(self._extract_key_terms(response))
        
        # Calculate overlap
        if prompt_terms:
            overlap = len(prompt_terms & response_terms) / len(prompt_terms)
            
            if overlap < 0.3:
                issues.append(PromptIssue(
                    issue_type=IssueType.MISSING_CONTEXT,
                    description="Response may not adequately address the prompt",
                    severity="high",
                    suggestion="Ensure response covers key prompt topics"
                ))
        
        return issues
    
    def _check_criteria_met(self, response: str, 
                           criteria: List[str]) -> List[PromptIssue]:
        """Check if response meets acceptance criteria."""
        issues = []
        response_lower = response.lower()
        
        for criterion in criteria:
            key_terms = self._extract_key_terms(criterion)
            found = any(term in response_lower for term in key_terms)
            
            if not found:
                issues.append(PromptIssue(
                    issue_type=IssueType.UNMET_CRITERIA,
                    description=f"Response doesn't address: {criterion[:100]}",
                    severity="high",
                    suggestion="Regenerate with explicit focus on this criterion"
                ))
        
        return issues
    
    def _check_code_quality(self, response: str) -> List[PromptIssue]:
        """Check code quality in response."""
        issues = []
        
        # Extract code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', response)
        
        for code in code_blocks:
            # Check for error handling
            if 'try' not in code and 'catch' not in code and 'except' not in code:
                if len(code) > 200:  # Only for substantial code
                    issues.append(PromptIssue(
                        issue_type=IssueType.POOR_STRUCTURE,
                        description="Code lacks error handling",
                        severity="medium",
                        suggestion="Add try-catch/except blocks"
                    ))
            
            # Check for comments
            comment_patterns = [r'#\s', r'//\s', r'/\*', r'"""', r"'''"]
            has_comments = any(re.search(p, code) for p in comment_patterns)
            
            if len(code) > 300 and not has_comments:
                issues.append(PromptIssue(
                    issue_type=IssueType.POOR_STRUCTURE,
                    description="Code lacks documentation comments",
                    severity="low",
                    suggestion="Add comments explaining logic"
                ))
        
        return issues
    
    def _apply_refinement(self, prompt: str, issue: PromptIssue) -> str:
        """Apply a refinement to address an issue."""
        if issue.issue_type == IssueType.AMBIGUITY:
            # Can't automatically fix ambiguity without more context
            return prompt
        
        if issue.issue_type == IssueType.HALLUCINATION:
            # Replace strong claims with softer language
            replacements = {
                'always': 'typically',
                'never': 'rarely',
                'guaranteed': 'expected',
                'impossible': 'unlikely',
                'definitely': 'likely',
            }
            for strong, soft in replacements.items():
                prompt = re.sub(rf'\b{strong}\b', soft, prompt, flags=re.IGNORECASE)
        
        return prompt
    
    def _has_good_structure(self, prompt: str) -> bool:
        """Check if prompt has good structure."""
        # Has sections or is short enough not to need them
        if len(prompt) < 100:
            return True
        
        structure_markers = ['##', '**', ':', '-', '1.', '•', '\n\n']
        return any(marker in prompt for marker in structure_markers)
    
    def _add_structure(self, prompt: str,
                      criteria: Optional[List[str]] = None,
                      context: Optional[Dict[str, Any]] = None) -> str:
        """Add structure to prompt."""
        sections = []
        
        sections.append("## Context")
        sections.append(prompt)
        sections.append("")
        
        if criteria:
            sections.append("## Acceptance Criteria")
            for i, c in enumerate(criteria, 1):
                sections.append(f"{i}. {c}")
            sections.append("")
        
        if context:
            sections.append("## Additional Context")
            for key, value in context.items():
                sections.append(f"- **{key}**: {value}")
            sections.append("")
        
        sections.append("## Requirements")
        sections.append("- Code should be well-documented")
        sections.append("- Include error handling")
        sections.append("- Follow best practices")
        
        return "\n".join(sections)
    
    def _is_testable(self, prompt: str) -> bool:
        """Check if prompt will lead to testable output."""
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in self.TESTABILITY_KEYWORDS)
    
    def _add_testability_hints(self, prompt: str) -> str:
        """Add testability hints to prompt."""
        hints = """

## Testing Requirements
- Code should be modular and testable
- Functions should have clear inputs and outputs
- Include example usage or test cases
- Document expected behavior for edge cases
"""
        return prompt + hints
    
    def _calculate_confidence(self, issues: List[PromptIssue]) -> float:
        """Calculate confidence score based on remaining issues."""
        if not issues:
            return 1.0
        
        # Weight by severity
        severity_weights = {'high': 0.3, 'medium': 0.15, 'low': 0.05}
        
        total_penalty = sum(
            severity_weights.get(issue.severity, 0.1)
            for issue in issues
        )
        
        return max(0.0, min(1.0, 1.0 - total_penalty))
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter and return unique terms
        terms = [w for w in words if w not in stop_words]
        return list(dict.fromkeys(terms))  # Preserve order, remove duplicates
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks."""
        return '```' in text or bool(re.search(r'def\s+\w+|function\s+\w+|class\s+\w+', text))
    
    def generate_feedback_for_coordinator(self, 
                                          refinement_results: List[RefinementResult]) -> Dict[str, Any]:
        """
        Generate feedback for the Coordinator agent.
        
        Args:
            refinement_results: List of refinement results
            
        Returns:
            Feedback dictionary for updating build plan
        """
        feedback = {
            "summary": {},
            "recommendations": [],
            "constraints": []
        }
        
        # Aggregate issues
        all_issues = []
        for result in refinement_results:
            all_issues.extend(result.issues_found)
        
        # Count by type
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.issue_type.value
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        feedback["summary"]["total_issues"] = len(all_issues)
        feedback["summary"]["by_type"] = issue_counts
        feedback["summary"]["avg_confidence"] = (
            sum(r.confidence_score for r in refinement_results) / len(refinement_results)
            if refinement_results else 0.0
        )
        
        # Generate recommendations
        if issue_counts.get(IssueType.HALLUCINATION.value, 0) > 2:
            feedback["recommendations"].append(
                "Add stricter validation for generated outputs"
            )
            feedback["constraints"].append(
                "require_output_validation: true"
            )
        
        if issue_counts.get(IssueType.AMBIGUITY.value, 0) > 3:
            feedback["recommendations"].append(
                "Require more specific acceptance criteria"
            )
        
        if issue_counts.get(IssueType.UNTESTABLE.value, 0) > 2:
            feedback["recommendations"].append(
                "Include test cases in code generation tasks"
            )
            feedback["constraints"].append(
                "require_tests: true"
            )
        
        return feedback
    
    def learn_from_result(self, prompt: str, response: str,
                         was_successful: bool, feedback: Optional[str] = None) -> None:
        """
        Learn from a prompt/response pair.
        
        Args:
            prompt: The original prompt
            response: The generated response
            was_successful: Whether the result was acceptable
            feedback: Optional human feedback
        """
        pattern = {
            "prompt_length": len(prompt),
            "response_length": len(response),
            "has_structure": self._has_good_structure(prompt),
            "is_testable": self._is_testable(prompt),
            "was_successful": was_successful,
            "feedback": feedback
        }
        
        # Extract features
        pattern["key_terms"] = self._extract_key_terms(prompt)[:10]
        
        self.learned_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.learned_patterns) > 100:
            self.learned_patterns = self.learned_patterns[-100:]
