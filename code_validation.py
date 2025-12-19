import ast
from typing import List, Dict, Optional, Tuple

class CodeValidator:
    def __init__(self, code: str):
        self.code = code
        self.tree: Optional[ast.AST] = None
        self.error_message: Optional[str] = None
        
        # Define security boundaries
        self.ALLOWED_IMPORTS = {'math', 'numpy', 'np', 'gym', 'gymnasium'}
        self.FORBIDDEN_FUNCS = {'eval', 'exec', 'open', 'input', 'system', 'subprocess'}
        self.FORBIDDEN_MODULES = {'os', 'sys', 'shutil', 'requests', 'socket'}

        try:
            self.tree = ast.parse(code)
        except SyntaxError as e:
            self.error_message = f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"

    def validate(self) -> Tuple[bool, str]:
        """Returns (is_valid, error_feedback_string)"""
        # 1. Check Syntax
        if self.tree is None:
            return False, self.error_message

        # 2. Check Security (Blacklist)
        checker = BlacklistChecker(self.FORBIDDEN_FUNCS, self.FORBIDDEN_MODULES, self.ALLOWED_IMPORTS)
        checker.visit(self.tree)
        
        if checker.violations:
            feedback = "Security Violation: The following are prohibited: " + ", ".join(checker.violations)
            return False, feedback

        return True, ""

# Updated Checker to include Whitelist logic
class BlacklistChecker(ast.NodeVisitor):
    def __init__(self, forbidden_funcs, forbidden_modules, allowed_imports):
        self.forbidden_funcs = forbidden_funcs
        self.forbidden_modules = forbidden_modules
        self.allowed_imports = allowed_imports
        self.violations = []
    
    def visit_Import(self, node):
        for alias in node.names:
            base = alias.name.split('.')[0]
            if base in self.forbidden_modules or base not in self.allowed_imports:
                self.violations.append(f"import '{alias.name}'")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.forbidden_funcs:
            self.violations.append(f"function '{node.func.id}()'")
        self.generic_visit(node)