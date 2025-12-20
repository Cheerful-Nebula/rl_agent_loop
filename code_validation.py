import ast
import numpy as np
import traceback
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

    def validate_static(self) -> Tuple[bool, str]:
        """Performs Static Analysis (Syntax + Security). Returns (is_valid, feedback)"""
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

    def validate_runtime(self) -> Tuple[bool, str]:
        """
        Performs Runtime Analysis (Execution Test).
        Actually runs the code with dummy data to catch logical crashes.
        Checks for crashes AND mathematical stability (NaN/Inf).
        Returns (is_valid, feedback)
        """
        # Create a restricted local scope
        local_scope = {}
        
        # 1. Define the function in the local scope
        try:
            exec(self.code, {"np": np, "math": __import__("math")}, local_scope)
        except Exception as e:
            return False, f"Runtime Definition Error: Code could not be loaded. {str(e)}"

        # 2. Check if function exists
        if "calculate_reward" not in local_scope:
            return False, "Runtime Error: Function 'calculate_reward' was not found in the generated code."

        # 3. Test with MULTIPLE Scenarios to catch edge cases
        # Scenario A: Standard Flight
        scenarios = [
            ("Standard", np.array([0.0, 1.0, 0.5, -0.5, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)),
            ("Zero Velocity (Div/0 Risk)", np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            ("Extreme Values (Overflow Risk)", np.array([100.0, 100.0, 50.0, 50.0, 3.14, 10.0, 1.0, 1.0], dtype=np.float32))
        ]

        func = local_scope["calculate_reward"]
        
        for name, obs in scenarios:
            try:
                # Mock inputs
                val = func(obs, 0.0, False, False, {"is_success": False})
                
                # --- MATH CHECK ---
                # 1. Check if it returns a number
                if not isinstance(val, (int, float, np.number)):
                     return False, f"Runtime Error ({name}): Function returned {type(val)}, expected float."
                
                # 2. Check for NaN or Infinity
                if not np.isfinite(val):
                    return False, f"Math Error ({name}): Reward function returned {val} (NaN/Infinity). Check for division by zero or log of negative numbers."

            except Exception as e:
                return False, f"Runtime Execution Error ({name}): {str(e)}"

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