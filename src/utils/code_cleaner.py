import re

class CodeCleaner:
    @staticmethod
    def clean(text: str) -> str:
        """
        Removes markdown code blocks, language identifiers, and 
        leading/trailing whitespace from AI-generated strings.
        """
        if not text:
            return ""

        # Pattern matches: ```[optional language] \n [code content] \n ```
        # It handles both ```python and just ```
        pattern = r"```(?:\w+)?\n([\s\S]*?)\n```|```([\s\S]*?)```"
        
        matches = re.findall(pattern, text)
        if matches:
            # findall with groups returns tuples, we take the first non-empty group
            code_parts = [m[0] or m[1] for m in matches]
            return "\n\n".join(code_parts).strip()

        # Fallback: if no markdown blocks found, just return the trimmed text
        # (The LLM might have actually followed instructions and returned pure code)
        return text.strip()

    @staticmethod
    def format_as_comment(text: str, language: str) -> str:
        """Helper to wrap text as a comment based on file type."""
        if language.lower() in ['python', 'py']:
            return f'"""\n{text}\n"""'
        return f'/*\n{text}\n*/'