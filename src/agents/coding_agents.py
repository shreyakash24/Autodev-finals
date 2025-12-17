"""
Coding Agents (Frontend, Backend, Database)
Generate code artifacts with documentation blocks and architectural comments.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import re

from crewai import Agent, Task as CrewTask

from ..models import GeneratedArtifact, AgentType, UserStory
from ..utils.llm_config import get_llm


class BaseCodingAgent:
    """Base class for all coding agents."""
    
    def __init__(self, agent_type: AgentType, role: str, goal: str, backstory: str):
        self.agent_type = agent_type
        self.llm = get_llm(temperature=0.2)
        
        # Create CrewAI agent if LLM is available
        self.crew_agent = None
        if self.llm:
            try:
                self.crew_agent = Agent(
                    role=role,
                    goal=goal,
                    backstory=backstory,
                    verbose=True,
                    allow_delegation=False,
                    llm=self.llm
                )
            except Exception as e:
                print(f"Warning: Failed to create CrewAI agent: {e}")
                print("Will use template-based generation instead.")
                self.crew_agent = None
    
    def _create_artifact(self, file_path: str, content: str, 
                        artifact_type: str, language: str,
                        documentation: str = "") -> GeneratedArtifact:
        """Create a GeneratedArtifact object."""
        return GeneratedArtifact(
            file_path=file_path,
            content=content,
            artifact_type=artifact_type,
            language=language,
            agent_type=self.agent_type,
            documentation=documentation
        )


class FrontendCodingAgent(BaseCodingAgent):
    """
    Frontend Coding Agent
    Generates React components, routes, forms, state management, and styling with accessibility.
    Provides component generation, routing, forms, and state management capabilities.
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.FRONTEND_CODER,
            role="Frontend Developer",
            goal="Generate high-quality React components with accessibility",
            backstory="""You are an expert React developer who creates clean,
            accessible, and well-documented frontend code. You follow best
            practices for component design and state management."""
        )
    
    def generate_component_scaffold(self, requirements: Dict[str, Any],
                                   tech_stack: str = "React") -> List[GeneratedArtifact]:
        """
        Generate React component scaffold.
        
        Args:
            requirements: Functional requirements
            tech_stack: Frontend framework (default: React)
            
        Returns:
            List of generated artifacts
        """
        # If LLM is available, use it to generate code from user stories
        if self.llm and self.crew_agent and requirements.get('user_stories'):
            return self._generate_from_stories(requirements, tech_stack)
        
        # Fallback to template-based generation if no LLM
        return self._generate_from_templates(requirements, tech_stack)
    
    def _generate_from_stories(self, requirements: Dict[str, Any], tech_stack: str) -> List[GeneratedArtifact]:
        """Generate components dynamically from user stories using LLM."""
        artifacts = []
        user_stories = requirements.get('user_stories', [])
        
        if not user_stories:
            return self._generate_from_templates(requirements, tech_stack)
        
        # Create a task for the agent to generate components
        from crewai import Task as CrewTask
        
        # Build prompt from user stories
        stories_text = "\n".join([
            f"Story {i+1}: {story.get('title', 'Untitled')}\n"
            f"Description: {story.get('description', 'No description')}\n"
            f"Acceptance Criteria: {', '.join(story.get('acceptance_criteria', []))}\n"
            for i, story in enumerate(user_stories)
        ])
        
        # Generate main App component
        app_task = CrewTask(
            description=f"""Generate a React TypeScript App component based on these user stories:
{stories_text}

Create a modern React application with:
1. Proper routing using React Router
2. Error boundaries
3. Accessibility features (skip links, ARIA labels)
4. Loading states

Return ONLY the complete TypeScript code for src/App.tsx, no explanations.""",
            agent=self.crew_agent,
            expected_output="Complete TypeScript React component code"
        )
        
        try:
            app_result = app_task.execute_sync()
            app_content = str(app_result) if app_result else self._generate_app_component(requirements)
            
            artifacts.append(self._create_artifact(
                "src/App.tsx",
                app_content,
                "component",
                "typescript",
                "Main application component with routing"
            ))
        except Exception as e:
            print(f"Error generating App component with LLM: {e}")
            # Fallback to template
            app_content = self._generate_app_component(requirements)
            artifacts.append(self._create_artifact(
                "src/App.tsx",
                app_content,
                "component",
                "typescript",
                "Main application component with routing"
            ))
        
        # Generate components for each user story
        for i, story in enumerate(user_stories[:5]):  # Limit to 5 stories
            story_title = story.get('title', f'Feature{i+1}')
            story_desc = story.get('description', '')
            acceptance_criteria = story.get('acceptance_criteria', [])
            
            component_task = CrewTask(
                description=f"""Generate a React TypeScript component for this user story:

Title: {story_title}
Description: {story_desc}
Acceptance Criteria:
{chr(10).join(f'- {criterion}' for criterion in acceptance_criteria)}

Create a complete, functional React component with:
1. TypeScript interfaces for props and state
2. Proper accessibility (ARIA labels, keyboard navigation)
3. Error handling
4. Loading states if needed
5. Form validation if it's a form component

Return ONLY the complete TypeScript component code, no explanations.""",
                agent=self.crew_agent,
                expected_output="Complete TypeScript React component code"
            )
            
            try:
                component_result = component_task.execute_sync()
                component_content = str(component_result) if component_result else self._generate_feature_component(story_title, [])
            except Exception as e:
                print(f"Error generating component for story '{story_title}': {e}")
                component_content = self._generate_feature_component(story_title, [])
            
            safe_name = self._to_pascal_case(story_title)
            artifacts.append(self._create_artifact(
                f"src/components/{safe_name}/{safe_name}.tsx",
                component_content,
                "component",
                "typescript",
                f"Component for {story_title}"
            ))
        
        return artifacts
    
    def _generate_from_templates(self, requirements: Dict[str, Any], tech_stack: str) -> List[GeneratedArtifact]:
        """Generate components from templates (fallback when LLM unavailable)."""
        artifacts = []
        
        # Generate main App component
        app_content = self._generate_app_component(requirements)
        artifacts.append(self._create_artifact(
            "src/App.tsx",
            app_content,
            "component",
            "typescript",
            "Main application component with routing"
        ))
        
        # Generate router configuration
        router_content = self._generate_router(requirements)
        artifacts.append(self._create_artifact(
            "src/router/index.tsx",
            router_content,
            "router",
            "typescript",
            "Application routing configuration"
        ))
        
        # Generate feature components based on requirements
        features = requirements.get('features', [])
        personas = requirements.get('personas', [])
        
        for feature in features[:5]:  # Limit to 5 main features
            component_content = self._generate_feature_component(feature, personas)
            safe_name = self._to_pascal_case(feature)
            artifacts.append(self._create_artifact(
                f"src/components/{safe_name}/{safe_name}.tsx",
                component_content,
                "component",
                "typescript",
                f"Component for {feature} feature"
            ))
        
        return artifacts
    
    def generate_forms(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """
        Generate form components and state management.
        
        Args:
            requirements: Functional requirements
            
        Returns:
            List of generated artifacts
        """
        artifacts = []
        
        # Generate form components
        form_content = self._generate_generic_form()
        artifacts.append(self._create_artifact(
            "src/components/Form/Form.tsx",
            form_content,
            "component",
            "typescript",
            "Reusable form component with validation"
        ))
        
        # Generate state management
        store_content = self._generate_state_store(requirements)
        artifacts.append(self._create_artifact(
            "src/store/index.ts",
            store_content,
            "store",
            "typescript",
            "Application state management"
        ))
        
        # Generate hooks
        hooks_content = self._generate_custom_hooks()
        artifacts.append(self._create_artifact(
            "src/hooks/useForm.ts",
            hooks_content,
            "hook",
            "typescript",
            "Custom form handling hook"
        ))
        
        return artifacts
    
    def add_accessibility(self, components: List[GeneratedArtifact]) -> List[GeneratedArtifact]:
        """
        Add accessibility features to components.
        
        Args:
            components: List of component artifacts
            
        Returns:
            Updated artifacts with accessibility
        """
        # In a real implementation, this would analyze and modify components
        # For now, we return accessibility utilities
        artifacts = []
        
        a11y_utils = '''/**
 * Accessibility Utilities
 * Provides WCAG 2.1 AA compliance helpers for React components.

// Screen reader only text
export const srOnly: React.CSSProperties = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: '0',
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0, 0, 0, 0)',
  whiteSpace: 'nowrap',
  border: '0',
};

// Focus trap hook
export function useFocusTrap(ref: React.RefObject<HTMLElement>) {
  React.useEffect(() => {
    const element = ref.current;
    if (!element) return;
    
    const focusableElements = element.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };
    
    element.addEventListener('keydown', handleKeyDown);
    return () => element.removeEventListener('keydown', handleKeyDown);
  }, [ref]);
}

// Announce to screen readers
export function announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
  const region = document.createElement('div');
  region.setAttribute('role', 'status');
  region.setAttribute('aria-live', priority);
  region.setAttribute('aria-atomic', 'true');
  Object.assign(region.style, srOnly);
  region.textContent = message;
  document.body.appendChild(region);
  setTimeout(() => document.body.removeChild(region), 1000);
}

// Skip link component
export const SkipLink: React.FC<{ href: string; children: React.ReactNode }> = ({ href, children }) => (
  <a
    href={href}
    className="skip-link"
    style={{
      position: 'absolute',
      left: '-9999px',
      top: 'auto',
      width: '1px',
      height: '1px',
      overflow: 'hidden',
    }}
    onFocus={(e) => {
      e.currentTarget.style.position = 'fixed';
      e.currentTarget.style.top = '0';
      e.currentTarget.style.left = '0';
      e.currentTarget.style.width = 'auto';
      e.currentTarget.style.height = 'auto';
      e.currentTarget.style.overflow = 'visible';
      e.currentTarget.style.zIndex = '9999';
      e.currentTarget.style.padding = '1rem';
      e.currentTarget.style.background = '#000';
      e.currentTarget.style.color = '#fff';
    }}
    onBlur={(e) => {
      Object.assign(e.currentTarget.style, srOnly);
    }}
  >
    {children}
  </a>
);
'''
        artifacts.append(self._create_artifact(
            "src/utils/accessibility.tsx",
            a11y_utils,
            "utility",
            "typescript",
            "Accessibility utilities for WCAG compliance"
        ))
        
        return artifacts
    
    def _generate_app_component(self, requirements: Dict[str, Any]) -> str:
        """Generate main App component."""
        return '''/**
 * Main Application Component
 * Component hierarchy: App > BrowserRouter > ErrorBoundary > AppRoutes

import React, { Suspense } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { AppRoutes } from './router';
import { ErrorBoundary } from './components/ErrorBoundary';
import { SkipLink } from './utils/accessibility';
import './styles/global.css';

const LoadingFallback: React.FC = () => (
  <div role="status" aria-label="Loading">
    <span className="sr-only">Loading...</span>
    <div className="loading-spinner" />
  </div>
);

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <SkipLink href="#main-content">Skip to main content</SkipLink>
        <Suspense fallback={<LoadingFallback />}>
          <main id="main-content" tabIndex={-1}>
            <AppRoutes />
          </main>
        </Suspense>
      </ErrorBoundary>
    </BrowserRouter>
  );
};

export default App;
'''
    
    def _generate_router(self, requirements: Dict[str, Any]) -> str:
        """Generate router configuration."""
        features = requirements.get('features', ['Home', 'Dashboard'])
        
        routes = []
        imports = []
        for feature in features[:5]:
            safe_name = self._to_pascal_case(feature)
            imports.append(f"const {safe_name} = lazy(() => import('../components/{safe_name}/{safe_name}'));")
            routes.append(f"  {{ path: '/{feature.lower().replace(' ', '-')}', element: <{safe_name} /> }},")
        
        return f'''/**
 * Application Router Configuration
 * Lazy-loaded routes for code splitting

import React, {{ lazy }} from 'react';
import {{ Routes, Route }} from 'react-router-dom';

// Lazy load components for code splitting
{chr(10).join(imports)}

const routes = [
  {{ path: '/', element: <Home /> }},
{chr(10).join(routes)}
];

export const AppRoutes: React.FC = () => (
  <Routes>
    {{routes.map((route) => (
      <Route key={{route.path}} path={{route.path}} element={{route.element}} />
    ))}}
    <Route path="*" element={{<NotFound />}} />
  </Routes>
);

const Home = lazy(() => import('../components/Home/Home'));
const NotFound = lazy(() => import('../components/NotFound/NotFound'));
'''
    
    def _generate_feature_component(self, feature: str, personas: List[str]) -> str:
        """Generate a feature component."""
        safe_name = self._to_pascal_case(feature)
        
        return f'''/**
 * {safe_name} Component
 * Feature: {feature}
 * Target Personas: {', '.join(personas) if personas else 'All users'}
 * Handles the {feature} functionality with proper accessibility and error handling.

import React, {{ useState, useCallback }} from 'react';
import {{ useForm }} from '../../hooks/useForm';
import {{ announce }} from '../../utils/accessibility';

interface {safe_name}Props {{
  className?: string;
}}

export const {safe_name}: React.FC<{safe_name}Props> = ({{ className = '' }}) => {{
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(async (data: Record<string, unknown>) => {{
    setIsLoading(true);
    setError(null);
    
    try {{
      // API call would go here
      announce('Operation completed successfully');
    }} catch (err) {{
      const message = err instanceof Error ? err.message : 'An error occurred';
      setError(message);
      announce(message, 'assertive');
    }} finally {{
      setIsLoading(false);
    }}
  }}, []);

  return (
    <section 
      className={{`{feature.lower().replace(' ', '-')}-container ${{className}}`}}
      aria-labelledby="{feature.lower().replace(' ', '-')}-heading"
    >
      <h1 id="{feature.lower().replace(' ', '-')}-heading">{feature}</h1>
      
      {{error && (
        <div role="alert" className="error-message">
          {{error}}
        </div>
      )}}
      
      {{isLoading ? (
        <div role="status" aria-label="Loading">
          <span className="sr-only">Loading...</span>
        </div>
      ) : (
        <div className="{feature.lower().replace(' ', '-')}-content">
          {{/* Feature content goes here */}}
        </div>
      )}}
    </section>
  );
}};

export default {safe_name};
'''
    
    def _generate_generic_form(self) -> str:
        """Generate a generic form component."""
        return '''/**
 * Generic Form Component
 * Reusable form with validation, accessibility, and error handling.
 * Uses controlled inputs with proper ARIA attributes.

import React, { FormEvent, ReactNode } from 'react';

interface FormField {
  name: string;
  label: string;
  type: 'text' | 'email' | 'password' | 'number' | 'textarea' | 'select';
  required?: boolean;
  options?: { value: string; label: string }[];
  validation?: (value: string) => string | null;
}

interface FormProps {
  fields: FormField[];
  onSubmit: (data: Record<string, string>) => void;
  submitLabel?: string;
  children?: ReactNode;
}

export const Form: React.FC<FormProps> = ({
  fields,
  onSubmit,
  submitLabel = 'Submit',
  children,
}) => {
  const [values, setValues] = React.useState<Record<string, string>>({});
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const [touched, setTouched] = React.useState<Record<string, boolean>>({});

  const handleChange = (name: string, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }));
    
    // Clear error on change
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };

  const handleBlur = (name: string) => {
    setTouched((prev) => ({ ...prev, [name]: true }));
    
    // Validate on blur
    const field = fields.find((f) => f.name === name);
    if (field?.validation) {
      const error = field.validation(values[name] || '');
      if (error) {
        setErrors((prev) => ({ ...prev, [name]: error }));
      }
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    
    // Validate all fields
    const newErrors: Record<string, string> = {};
    let hasErrors = false;
    
    for (const field of fields) {
      if (field.required && !values[field.name]) {
        newErrors[field.name] = `${field.label} is required`;
        hasErrors = true;
      } else if (field.validation) {
        const error = field.validation(values[field.name] || '');
        if (error) {
          newErrors[field.name] = error;
          hasErrors = true;
        }
      }
    }
    
    if (hasErrors) {
      setErrors(newErrors);
      // Focus first error field
      const firstErrorField = fields.find((f) => newErrors[f.name]);
      if (firstErrorField) {
        document.getElementById(firstErrorField.name)?.focus();
      }
      return;
    }
    
    onSubmit(values);
  };

  const renderField = (field: FormField) => {
    const hasError = touched[field.name] && errors[field.name];
    const inputId = field.name;
    const errorId = `${field.name}-error`;
    
    const commonProps = {
      id: inputId,
      name: field.name,
      value: values[field.name] || '',
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
        handleChange(field.name, e.target.value),
      onBlur: () => handleBlur(field.name),
      'aria-invalid': hasError ? true : undefined,
      'aria-describedby': hasError ? errorId : undefined,
      required: field.required,
    };

    return (
      <div key={field.name} className="form-field">
        <label htmlFor={inputId}>
          {field.label}
          {field.required && <span aria-hidden="true"> *</span>}
        </label>
        
        {field.type === 'textarea' ? (
          <textarea {...commonProps} />
        ) : field.type === 'select' ? (
          <select {...commonProps}>
            <option value="">Select {field.label}</option>
            {field.options?.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        ) : (
          <input type={field.type} {...commonProps} />
        )}
        
        {hasError && (
          <span id={errorId} role="alert" className="field-error">
            {errors[field.name]}
          </span>
        )}
      </div>
    );
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      {fields.map(renderField)}
      {children}
      <button type="submit">{submitLabel}</button>
    </form>
  );
};

export default Form;
'''
    
    def _generate_state_store(self, requirements: Dict[str, Any]) -> str:
        """Generate state management store."""
        return '''/**
 * Application State Store
 * Uses React Context + useReducer for lightweight state management.
 * Manages global state including user, data, and UI state.

import React, { createContext, useContext, useReducer, ReactNode } from 'react';

// State Types
interface AppState {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  data: Record<string, unknown>;
}

interface User {
  id: string;
  name: string;
  email: string;
}

// Action Types
type Action =
  | { type: 'SET_USER'; payload: User | null }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_DATA'; payload: { key: string; value: unknown } }
  | { type: 'RESET' };

// Initial State
const initialState: AppState = {
  user: null,
  isLoading: false,
  error: null,
  data: {},
};

// Reducer
function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_USER':
      return { ...state, user: action.payload, error: null };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'SET_DATA':
      return {
        ...state,
        data: { ...state.data, [action.payload.key]: action.payload.value },
      };
    case 'RESET':
      return initialState;
    default:
      return state;
  }
}

// Context
interface StoreContextType {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}

const StoreContext = createContext<StoreContextType | undefined>(undefined);

// Provider Component
export const StoreProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <StoreContext.Provider value={{ state, dispatch }}>
      {children}
    </StoreContext.Provider>
  );
};

// Custom Hook
export function useStore() {
  const context = useContext(StoreContext);
  if (!context) {
    throw new Error('useStore must be used within a StoreProvider');
  }
  return context;
}

// Action Creators
export const actions = {
  setUser: (user: User | null): Action => ({ type: 'SET_USER', payload: user }),
  setLoading: (loading: boolean): Action => ({ type: 'SET_LOADING', payload: loading }),
  setError: (error: string | null): Action => ({ type: 'SET_ERROR', payload: error }),
  setData: (key: string, value: unknown): Action => ({
    type: 'SET_DATA',
    payload: { key, value },
  }),
  reset: (): Action => ({ type: 'RESET' }),
};
'''
    
    def _generate_custom_hooks(self) -> str:
        """Generate custom hooks."""
        return '''/**
 * Custom Form Hook
 * Provides form state management with validation and submission handling.

import { useState, useCallback, ChangeEvent, FormEvent } from 'react';

interface UseFormOptions<T> {
  initialValues: T;
  validate?: (values: T) => Partial<Record<keyof T, string>>;
  onSubmit: (values: T) => void | Promise<void>;
}

interface UseFormReturn<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isSubmitting: boolean;
  handleChange: (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => void;
  handleBlur: (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => void;
  handleSubmit: (e: FormEvent) => void;
  setFieldValue: (field: keyof T, value: T[keyof T]) => void;
  setFieldError: (field: keyof T, error: string) => void;
  resetForm: () => void;
}

export function useForm<T extends Record<string, unknown>>({
  initialValues,
  validate,
  onSubmit,
}: UseFormOptions<T>): UseFormReturn<T> {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
      const { name, value, type } = e.target;
      const newValue = type === 'checkbox' ? (e.target as HTMLInputElement).checked : value;
      
      setValues((prev) => ({ ...prev, [name]: newValue }));
      
      // Clear error on change
      if (errors[name as keyof T]) {
        setErrors((prev) => ({ ...prev, [name]: undefined }));
      }
    },
    [errors]
  );

  const handleBlur = useCallback(
    (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
      const { name } = e.target;
      setTouched((prev) => ({ ...prev, [name]: true }));
      
      // Validate on blur
      if (validate) {
        const validationErrors = validate(values);
        if (validationErrors[name as keyof T]) {
          setErrors((prev) => ({ ...prev, [name]: validationErrors[name as keyof T] }));
        }
      }
    },
    [values, validate]
  );

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      
      // Mark all fields as touched
      const allTouched = Object.keys(values).reduce(
        (acc, key) => ({ ...acc, [key]: true }),
        {} as Partial<Record<keyof T, boolean>>
      );
      setTouched(allTouched);
      
      // Validate
      if (validate) {
        const validationErrors = validate(values);
        if (Object.keys(validationErrors).length > 0) {
          setErrors(validationErrors);
          return;
        }
      }
      
      setIsSubmitting(true);
      try {
        await onSubmit(values);
      } finally {
        setIsSubmitting(false);
      }
    },
    [values, validate, onSubmit]
  );

  const setFieldValue = useCallback((field: keyof T, value: T[keyof T]) => {
    setValues((prev) => ({ ...prev, [field]: value }));
  }, []);

  const setFieldError = useCallback((field: keyof T, error: string) => {
    setErrors((prev) => ({ ...prev, [field]: error }));
  }, []);

  const resetForm = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  }, [initialValues]);

  return {
    values,
    errors,
    touched,
    isSubmitting,
    handleChange,
    handleBlur,
    handleSubmit,
    setFieldValue,
    setFieldError,
    resetForm,
  };
}

export default useForm;
'''
    
    def _to_pascal_case(self, text: str) -> str:
        """Convert text to PascalCase."""
        # Remove special characters and split by spaces or underscores
        words = re.sub(r'[^a-zA-Z0-9\s_]', '', text).replace('_', ' ').split()
        return ''.join(word.capitalize() for word in words)


class BackendCodingAgent(BaseCodingAgent):
    """
    Backend Coding Agent
    Generates REST endpoints in Python (FastAPI), services, controllers, and validation.
    Provides API generation with endpoints, services, controllers and validation layer.
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.BACKEND_CODER,
            role="Backend Developer",
            goal="Generate clean, secure REST APIs with proper validation",
            backstory="""You are an expert backend developer who creates
            well-structured APIs with proper error handling, validation,
            and security practices."""
        )
    
    def generate_api_contracts(self, requirements: Dict[str, Any],
                               tech_stack: str = "FastAPI") -> List[GeneratedArtifact]:
        """
        Generate API contracts and endpoints.
        
        Args:
            requirements: Functional requirements
            tech_stack: Backend framework
            
        Returns:
            List of generated artifacts
        """
        # If LLM is available, use it to generate code from user stories
        if self.llm and self.crew_agent and requirements.get('user_stories'):
            return self._generate_backend_from_stories(requirements, tech_stack)
        
        # Fallback to template-based generation
        return self._generate_backend_from_templates(requirements, tech_stack)
    
    def _generate_backend_from_stories(self, requirements: Dict[str, Any], tech_stack: str) -> List[GeneratedArtifact]:
        """Generate backend code dynamically from user stories using LLM."""
        artifacts = []
        user_stories = requirements.get('user_stories', [])
        
        if not user_stories:
            return self._generate_backend_from_templates(requirements, tech_stack)
        
        from crewai import Task as CrewTask
        
        # Build prompt from user stories
        stories_text = "\n".join([
            f"Story {i+1}: {story.get('title', 'Untitled')}\n"
            f"Description: {story.get('description', 'No description')}\n"
            f"Acceptance Criteria: {', '.join(story.get('acceptance_criteria', []))}\n"
            for i, story in enumerate(user_stories)
        ])
        
        # Generate API routes based on user stories
        routes_task = CrewTask(
            description=f"""Generate FastAPI routes based on these user stories:
{stories_text}

Create RESTful API endpoints with:
1. Proper HTTP methods (GET, POST, PUT, DELETE)
2. Pydantic models for request/response validation
3. Error handling with appropriate status codes
4. Async/await for database operations
5. Proper documentation strings

Return ONLY the complete Python code for app/api/routes.py, no explanations.""",
            agent=self.crew_agent,
            expected_output="Complete Python FastAPI routes code"
        )
        
        try:
            routes_result = routes_task.execute_sync()
            print("LLM output:", routes_result)
            routes_content = str(routes_result) if routes_result else self._generate_routes(requirements)
            
            artifacts.append(self._create_artifact(
                "app/api/routes.py",
                routes_content,
                "routes",
                "python",
                "API route definitions"
            ))
        except Exception as e:
            print(f"Error generating routes with LLM: {e}")
            routes_content = self._generate_routes(requirements)
            artifacts.append(self._create_artifact(
                "app/api/routes.py",
                routes_content,
                "routes",
                "python",
                "API route definitions"
            ))
        
        # Generate schemas
        schemas_task = CrewTask(
            description=f"""Generate Pydantic schemas for these user stories:
{stories_text}

Create Pydantic models for:
1. Request bodies
2. Response models
3. Database models (SQLAlchemy)
4. Proper field validation
5. Example values

Return ONLY the complete Python code for app/schemas.py, no explanations.""",
            agent=self.crew_agent,
            expected_output="Complete Python Pydantic schemas code"
        )
        
        try:
            schemas_result = schemas_task.execute_sync()
            schemas_content = str(schemas_result) if schemas_result else self._generate_schemas(requirements)
            
            artifacts.append(self._create_artifact(
                "app/schemas.py",
                schemas_content,
                "schemas",
                "python",
                "Pydantic schemas for validation"
            ))
        except Exception as e:
            print(f"Error generating schemas with LLM: {e}")
            schemas_content = self._generate_schemas(requirements)
            artifacts.append(self._create_artifact(
                "app/schemas.py",
                schemas_content,
                "schemas",
                "python",
                "Pydantic schemas for validation"
            ))
        
        # Always generate main app (not story-specific)
        main_content = self._generate_main_app()
        artifacts.append(self._create_artifact(
            "app/main.py",
            main_content,
            "application",
            "python",
            "Main FastAPI application"
        ))
        
        return artifacts
    
    def _generate_backend_from_templates(self, requirements: Dict[str, Any], tech_stack: str) -> List[GeneratedArtifact]:
        """Generate backend code from templates (fallback)."""
        artifacts = []
        
        # Generate main app
        main_content = self._generate_main_app()
        artifacts.append(self._create_artifact(
            "app/main.py",
            main_content,
            "application",
            "python",
            "Main FastAPI application"
        ))
        
        # Generate API routes
        routes_content = self._generate_routes(requirements)
        artifacts.append(self._create_artifact(
            "app/api/routes.py",
            routes_content,
            "routes",
            "python",
            "API route definitions"
        ))
        
        # Generate schemas
        schemas_content = self._generate_schemas(requirements)
        artifacts.append(self._create_artifact(
            "app/schemas.py",
            schemas_content,
            "schemas",
            "python",
            "Pydantic schemas for validation"
        ))
        
        return artifacts
    
    def generate_services(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """
        Generate service layer with business logic.
        
        Args:
            requirements: Functional requirements
            
        Returns:
            List of generated artifacts
        """
        artifacts = []
        
        service_content = self._generate_base_service()
        artifacts.append(self._create_artifact(
            "app/services/base.py",
            service_content,
            "service",
            "python",
            "Base service with common operations"
        ))
        
        return artifacts
    
    def generate_controllers(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """
        Generate controllers with request handling.
        
        Args:
            requirements: Functional requirements
            
        Returns:
            List of generated artifacts
        """
        artifacts = []
        
        controller_content = self._generate_controller()
        artifacts.append(self._create_artifact(
            "app/controllers/base.py",
            controller_content,
            "controller",
            "python",
            "Base controller with validation"
        ))
        
        return artifacts
    
    def _generate_main_app(self) -> str:
        """Generate main FastAPI application."""
        return '''"""
Main FastAPI Application

Includes middleware (CORS, logging, error handler) and API routes.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="Generated API",
    description="API generated by the Agentic Code Generator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {process_time:.3f}s "
        f"status={response.status_code}"
    )
    
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}
'''
    
    def _generate_routes(self, requirements: Dict[str, Any]) -> str:
        """Generate API routes."""
        features = requirements.get('features', ['items'])
        
        return '''"""
API Routes

Routes are organized by resource type with CRUD operations.
Each route includes proper validation and error handling.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from app.schemas import ItemCreate, ItemUpdate, ItemResponse, PaginatedResponse
from app.services.base import BaseService

router = APIRouter()
service = BaseService()


@router.get("/items", response_model=PaginatedResponse[ItemResponse])
async def list_items(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
    search: Optional[str] = Query(None, description="Search term")
):
    """
    List all items with pagination.
    
    - **skip**: Number of items to skip (for pagination)
    - **limit**: Maximum number of items to return
    - **search**: Optional search term
    """
    items, total = await service.list_items(skip=skip, limit=limit, search=search)
    return PaginatedResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str):
    """
    Get a specific item by ID.
    
    - **item_id**: The unique identifier of the item
    """
    item = await service.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate):
    """
    Create a new item.
    
    - **item**: Item data to create
    """
    return await service.create_item(item)


@router.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: str, item: ItemUpdate):
    """
    Update an existing item.
    
    - **item_id**: The unique identifier of the item
    - **item**: Updated item data
    """
    updated = await service.update_item(item_id, item)
    if not updated:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated


@router.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: str):
    """
    Delete an item.
    
    - **item_id**: The unique identifier of the item to delete
    """
    deleted = await service.delete_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")
'''
    
    def _generate_schemas(self, requirements: Dict[str, Any]) -> str:
        """Generate Pydantic schemas."""
        return '''"""
Pydantic Schemas for Request/Response Validation

Schemas define the structure and validation rules for API data.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Generic, TypeVar
from datetime import datetime


# Generic type for paginated responses
T = TypeVar('T')


class ItemBase(BaseModel):
    """Base schema for items."""
    name: str = Field(..., min_length=1, max_length=100, description="Item name")
    description: Optional[str] = Field(None, max_length=500, description="Item description")
    status: str = Field(default="active", description="Item status")
    

class ItemCreate(ItemBase):
    """Schema for creating items."""
    pass


class ItemUpdate(BaseModel):
    """Schema for updating items (all fields optional)."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    status: Optional[str] = None


class ItemResponse(ItemBase):
    """Schema for item responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response schema."""
    items: List[T]
    total: int = Field(..., ge=0, description="Total number of items")
    skip: int = Field(..., ge=0, description="Number of items skipped")
    limit: int = Field(..., ge=1, description="Maximum items returned")
    
    @property
    def has_more(self) -> bool:
        """Check if there are more items."""
        return self.skip + len(self.items) < self.total


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
'''
    
    def _generate_base_service(self) -> str:
        """Generate base service."""
        return '''"""
Base Service Layer

Provides structured implementation.
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import uuid
from app.schemas import ItemCreate, ItemUpdate, ItemResponse


class BaseService:
    """
    Base service with common CRUD operations.
    In production, this would integrate with a database.
    """
    
    def __init__(self):
        # In-memory storage for demonstration
        self._items: Dict[str, Dict[str, Any]] = {}
    
    async def list_items(
        self,
        skip: int = 0,
        limit: int = 10,
        search: Optional[str] = None
    ) -> Tuple[List[ItemResponse], int]:
        """
        List items with pagination and optional search.
        
        Args:
            skip: Number of items to skip
            limit: Maximum items to return
            search: Optional search term
            
        Returns:
            Tuple of (items, total_count)
        """
        items = list(self._items.values())
        
        # Filter by search term
        if search:
            search_lower = search.lower()
            items = [
                i for i in items 
                if search_lower in i['name'].lower() or 
                   (i.get('description') and search_lower in i['description'].lower())
            ]
        
        total = len(items)
        items = items[skip:skip + limit]
        
        return [ItemResponse(**item) for item in items], total
    
    async def get_item(self, item_id: str) -> Optional[ItemResponse]:
        """
        Get a single item by ID.
        
        Args:
            item_id: Unique identifier
            
        Returns:
            Item if found, None otherwise
        """
        item = self._items.get(item_id)
        if item:
            return ItemResponse(**item)
        return None
    
    async def create_item(self, data: ItemCreate) -> ItemResponse:
        """
        Create a new item.
        
        Args:
            data: Item creation data
            
        Returns:
            Created item
        """
        item_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        item = {
            "id": item_id,
            **data.model_dump(),
            "created_at": now,
            "updated_at": now
        }
        
        self._items[item_id] = item
        return ItemResponse(**item)
    
    async def update_item(self, item_id: str, data: ItemUpdate) -> Optional[ItemResponse]:
        """
        Update an existing item.
        
        Args:
            item_id: Unique identifier
            data: Update data
            
        Returns:
            Updated item if found, None otherwise
        """
        if item_id not in self._items:
            return None
        
        item = self._items[item_id]
        update_data = data.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if value is not None:
                item[field] = value
        
        item["updated_at"] = datetime.utcnow()
        
        return ItemResponse(**item)
    
    async def delete_item(self, item_id: str) -> bool:
        """
        Delete an item.
        
        Args:
            item_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False
'''
    
    def _generate_controller(self) -> str:
        """Generate controller."""
        return '''"""
Base Controller

Controllers handle request processing and response formatting.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException


class BaseController:
    """
    Base controller with common request handling logic.
    """
    
    @staticmethod
    def validate_id(item_id: str) -> str:
        """
        Validate that an ID is properly formatted.
        
        Args:
            item_id: ID to validate
            
        Returns:
            Validated ID
            
        Raises:
            HTTPException: If ID is invalid
        """
        if not item_id or len(item_id) < 1:
            raise HTTPException(status_code=400, detail="Invalid ID format")
        return item_id
    
    @staticmethod
    def format_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Format a successful response.
        
        Args:
            data: Response data
            message: Optional message
            
        Returns:
            Formatted response dictionary
        """
        response = {"data": data, "success": True}
        if message:
            response["message"] = message
        return response
    
    @staticmethod
    def format_error(message: str, code: Optional[str] = None) -> Dict[str, Any]:
        """
        Format an error response.
        
        Args:
            message: Error message
            code: Optional error code
            
        Returns:
            Formatted error dictionary
        """
        error = {"detail": message, "success": False}
        if code:
            error["code"] = code
        return error
'''


class DatabaseCodingAgent(BaseCodingAgent):
    """
    Database Coding Agent
    Generates schema migrations and ORM models aligned with domain objects.
    
    Provides structured implementation.
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.DATABASE_CODER,
            role="Database Developer",
            goal="Generate optimized database schemas and ORM models",
            backstory="""You are an expert database developer who designs
            efficient schemas with proper indexing, relationships, and
            data integrity constraints."""
        )
    
    def generate_schema(self, requirements: Dict[str, Any],
                       tech_stack: str = "PostgreSQL") -> List[GeneratedArtifact]:
        """
        Generate database schema and migrations.
        
        Args:
            requirements: Functional requirements
            tech_stack: Database technology
            
        Returns:
            List of generated artifacts
        """
        # If LLM is available, use it to generate schema from user stories
        if self.llm and self.crew_agent and requirements.get('user_stories'):
            return self._generate_schema_from_stories(requirements, tech_stack)
        
        # Fallback to template-based generation
        return self._generate_schema_from_templates(requirements, tech_stack)
    
    def _generate_schema_from_stories(self, requirements: Dict[str, Any], tech_stack: str) -> List[GeneratedArtifact]:
        """Generate database schema dynamically from user stories using LLM."""
        artifacts = []
        user_stories = requirements.get('user_stories', [])
        
        if not user_stories:
            return self._generate_schema_from_templates(requirements, tech_stack)
        
        from crewai import Task as CrewTask
        
        # Build prompt from user stories
        stories_text = "\n".join([
            f"Story {i+1}: {story.get('title', 'Untitled')}\n"
            f"Description: {story.get('description', 'No description')}\n"
            f"Acceptance Criteria: {', '.join(story.get('acceptance_criteria', []))}\n"
            for i, story in enumerate(user_stories)
        ])
        
        # Generate SQLAlchemy models based on user stories
        models_task = CrewTask(
            description=f"""Generate SQLAlchemy models based on these user stories:
{stories_text}

Create database models with:
1. Proper table names and relationships
2. Field types based on requirements
3. Indexes for performance
4. Foreign keys and constraints
5. Timestamps and soft delete support

Return ONLY the complete Python code for app/models/base.py, no explanations.""",
            agent=self.crew_agent,
            expected_output="Complete Python SQLAlchemy models code"
        )
        
        try:
            models_result = models_task.execute_sync()
            models_content = str(models_result) if models_result else self._generate_models(requirements)
            
            artifacts.append(self._create_artifact(
                "app/models/base.py",
                models_content,
                "model",
                "python",
                "SQLAlchemy base model and mixins"
            ))
            print("LLM output:", models_result)
        except Exception as e:
            print(f"Error generating models with LLM: {e}")
            models_content = self._generate_models(requirements)
            artifacts.append(self._create_artifact(
                "app/models/base.py",
                models_content,
                "model",
                "python",
                "SQLAlchemy base model and mixins"
            ))
        
        # Generate migration (not story-specific, use template)
        migration_content = self._generate_migration(requirements)
        artifacts.append(self._create_artifact(
            "migrations/versions/001_initial.py",
            migration_content,
            "migration",
            "python",
            "Initial database migration"
        ))
        
        # Generate database config (not story-specific, use template)
        config_content = self._generate_db_config()
        artifacts.append(self._create_artifact(
            "app/database.py",
            config_content,
            "config",
            "python",
            "Database configuration and session management"
        ))
        
        return artifacts
    
    def _generate_schema_from_templates(self, requirements: Dict[str, Any], tech_stack: str) -> List[GeneratedArtifact]:
        """Generate database schema from templates (fallback)."""
        artifacts = []
        
        # Generate SQLAlchemy models
        models_content = self._generate_models(requirements)
        artifacts.append(self._create_artifact(
            "app/models/base.py",
            models_content,
            "model",
            "python",
            "SQLAlchemy base model and mixins"
        ))
        
        # Generate migration
        migration_content = self._generate_migration(requirements)
        artifacts.append(self._create_artifact(
            "migrations/versions/001_initial.py",
            migration_content,
            "migration",
            "python",
            "Initial database migration"
        ))
        
        # Generate database config
        config_content = self._generate_db_config()
        artifacts.append(self._create_artifact(
            "app/database.py",
            config_content,
            "config",
            "python",
            "Database configuration and session management"
        ))
        
        return artifacts
    
    def generate_orm_models(self, requirements: Dict[str, Any]) -> List[GeneratedArtifact]:
        """
        Generate ORM models aligned with domain objects.
        
        Args:
            requirements: Functional requirements
            
        Returns:
            List of generated artifacts
        """
        artifacts = []
        
        item_model = self._generate_item_model()
        artifacts.append(self._create_artifact(
            "app/models/item.py",
            item_model,
            "model",
            "python",
            "Item domain model"
        ))
        
        return artifacts
    
    def _generate_models(self, requirements: Dict[str, Any]) -> str:
        """Generate SQLAlchemy base models."""
        return '''"""
SQLAlchemy Base Models and Mixins

Provides structured implementation.
"""

from datetime import datetime
from typing import Any
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, declared_attr
import uuid

Base = declarative_base()


class IDMixin:
    """Mixin that adds a UUID primary key."""
    
    @declared_attr
    def id(cls) -> Column:
        return Column(
            String(36),
            primary_key=True,
            default=lambda: str(uuid.uuid4()),
            nullable=False
        )


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamps."""
    
    @declared_attr
    def created_at(cls) -> Column:
        return Column(
            DateTime,
            default=datetime.utcnow,
            nullable=False
        )
    
    @declared_attr
    def updated_at(cls) -> Column:
        return Column(
            DateTime,
            default=datetime.utcnow,
            onupdate=datetime.utcnow,
            nullable=False
        )


class SoftDeleteMixin:
    """Mixin that adds soft delete capability."""
    
    @declared_attr
    def is_deleted(cls) -> Column:
        return Column(Boolean, default=False, nullable=False)
    
    @declared_attr
    def deleted_at(cls) -> Column:
        return Column(DateTime, nullable=True)
    
    def soft_delete(self) -> None:
        """Mark the record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()


class BaseModel(Base, IDMixin, TimestampMixin):
    """
    Base model with common fields and methods.
    All domain models should inherit from this class.
    """
    __abstract__ = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update(self, **kwargs: Any) -> None:
        """Update model attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
'''
    
    def _generate_migration(self, requirements: Dict[str, Any]) -> str:
        """Generate database migration."""
        return '''"""
Initial Database Migration

Revision ID: 001
Create Date: 2024-01-01

This migration creates the initial database schema with
proper indexes and constraints.
"""

from alembic import op
import sqlalchemy as sa


# Revision identifiers
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial tables."""
    
    # Items table
    op.create_table(
        'items',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('is_deleted', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('deleted_at', sa.DateTime, nullable=True),
    )
    
    # Create indexes
    op.create_index('ix_items_name', 'items', ['name'])
    op.create_index('ix_items_status', 'items', ['status'])
    op.create_index('ix_items_created_at', 'items', ['created_at'])
    
    # Users table (if needed)
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    
    op.create_index('ix_users_email', 'users', ['email'], unique=True)


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('users')
    op.drop_table('items')
'''
    
    def _generate_db_config(self) -> str:
        """Generate database configuration."""
        return '''"""
Database Configuration

Provides structured implementation.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Database URL from environment
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'sqlite:///./app.db'  # Default to SQLite for development
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,
    max_overflow=10,
    echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    Use with FastAPI's Depends().
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Use when not using FastAPI dependencies.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
'''
    
    def _generate_item_model(self) -> str:
        """Generate item domain model."""
        return '''"""
Item Domain Model

Domain model for items with all business logic and relationships.
"""

from sqlalchemy import Column, String, Text, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.models.base import BaseModel, SoftDeleteMixin


class Item(BaseModel, SoftDeleteMixin):
    """
    Item domain model.
    
    Represents a generic item in the system with:
    - Basic attributes (name, description, status)
    - Timestamps (created_at, updated_at)
    - Soft delete capability
    """
    __tablename__ = 'items'
    
    # Attributes
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default='active', index=True)
    
    def __repr__(self) -> str:
        return f"<Item(id={self.id}, name={self.name})>"
    
    def activate(self) -> None:
        """Set item status to active."""
        self.status = 'active'
    
    def deactivate(self) -> None:
        """Set item status to inactive."""
        self.status = 'inactive'
    
    @property
    def is_active(self) -> bool:
        """Check if item is active."""
        return self.status == 'active' and not self.is_deleted
'''
