#!/usr/bin/env python3
"""
Main entry point for the Agentic Code Generation System
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """Run the Flask web server with UI."""
    from src.ui.app import run_server as start_server
    print(f"Starting Agentic Code Generator server at http://{host}:{port}")
    start_server(host=host, port=port, debug=debug)


def run_cli():
    """Run in CLI mode for direct code generation."""
    from src.crew import create_crew
    
    crew = create_crew()
    
    print("=" * 60)
    print("  Agentic Code Generation System - CLI Mode")
    print("=" * 60)
    print()
    
    # Interactive menu
    while True:
        print("\nOptions:")
        print("1. Parse requirements (JSON)")
        print("2. Parse requirements (CSV)")
        print("3. Build pipeline")
        print("4. Execute pipeline")
        print("5. Run tests")
        print("6. Analyze legacy repo")
        print("7. Refine prompt")
        print("8. View artifacts")
        print("9. Check quality gate")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            print("Enter JSON (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            data = '\n'.join(lines)
            
            try:
                spec = crew.process_requirements(data, 'json')
                print(f"\n✅ Parsed {len(spec.user_stories)} user stories")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '2':
            print("Enter CSV (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            data = '\n'.join(lines)
            
            try:
                spec = crew.process_requirements(data, 'csv')
                print(f"\n✅ Parsed {len(spec.user_stories)} user stories")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '3':
            try:
                pipeline = crew.build_pipeline()
                print(f"\n✅ Built pipeline with {len(pipeline.tasks)} tasks")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '4':
            try:
                print("\nExecuting pipeline...")
                pipeline = crew.execute_pipeline()
                completed = sum(1 for t in pipeline.tasks if t.status.value == 'completed')
                print(f"\n✅ Completed {completed}/{len(pipeline.tasks)} tasks")
                print(f"📄 Generated {len(pipeline.artifacts)} artifacts")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '5':
            test_type = input("Test type (unit/integration/e2e/all): ").strip() or 'all'
            try:
                results = crew.run_tests(test_type)
                for ttype, report in results.items():
                    print(f"\n{ttype.upper()} Tests:")
                    print(f"  Passed: {report['passed_tests']}/{report['total_tests']}")
                    print(f"  Coverage: {report['overall_coverage']:.1f}%")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '6':
            repo_path = input("Repository path: ").strip()
            if repo_path:
                try:
                    analysis = crew.analyze_legacy(repo_path)
                    print(f"\n✅ Analysis complete")
                    print(f"  Tech Stack: {analysis.get('tech_stack', {})}")
                    print(f"  Architecture: {analysis.get('architecture', 'Unknown')}")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
        
        elif choice == '7':
            print("Enter prompt (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            prompt = '\n'.join(lines)
            
            try:
                result = crew.refine_prompt(prompt)
                print(f"\n✅ Refined prompt (confidence: {result['confidence']:.1%})")
                print(f"\nImprovements:")
                for imp in result['improvements']:
                    print(f"  - {imp}")
                print(f"\nRefined prompt:\n{result['refined']}")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '8':
            artifacts = crew.get_artifacts()
            if artifacts:
                print(f"\n📄 {len(artifacts)} artifacts:")
                for a in artifacts:
                    print(f"  - {a['file_path']} ({a['language']})")
            else:
                print("\nNo artifacts generated yet")
        
        elif choice == '9':
            try:
                gate = crew.check_quality_gate()
                status = "✅ PASSED" if gate['passed'] else "❌ FAILED"
                print(f"\nQuality Gate: {status}")
                print(f"  Coverage Met: {gate['coverage_met']}")
                print(f"  Tests Passed: {gate['tests_passed']}")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        elif choice == '0':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid option")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Agentic Code Generation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py server              # Start web server
  python main.py server --port 8080  # Start on different port
  python main.py cli                 # Interactive CLI mode
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start web server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    server_parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    
    # CLI command
    subparsers.add_parser('cli', help='Interactive CLI mode')
    
    args = parser.parse_args()
    
    if args.command == 'server':
        run_server(
            host=args.host,
            port=args.port,
            debug=not args.no_debug
        )
    elif args.command == 'cli':
        run_cli()
    else:
        # Default to server
        print("No command specified. Starting server...")
        run_server()


if __name__ == '__main__':
    main()
