import ast
import numpy as np
from typing import Tuple, Optional, Dict, Any

class CodeValidator:
    def __init__(self, code: str):
        self.code = code
        self.tree: Optional[ast.AST] = None
        self.error_message: Optional[str] = None
        
        # Security: Whitelist allowed libraries
        self.ALLOWED_IMPORTS = {'math', 'numpy', 'np', 'gym', 'gymnasium', 'typing'}
        self.FORBIDDEN_FUNCS = {'eval', 'exec', 'open', 'input', 'system', 'subprocess'}
        self.FORBIDDEN_MODULES = {'os', 'sys', 'shutil', 'requests', 'socket'}

        try:
            self.tree = ast.parse(code)
        except SyntaxError as e:
            self.error_message = f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"

    def validate(self) -> Tuple[bool, str]:
        """
        Main validation entry point. 
        Returns (is_valid: bool, feedback_or_code: str)
        """
        # 1. Syntax Check
        if self.tree is None:
            return False, self.error_message
        
        # 2. Static Security Check
        checker = BlacklistChecker(self.FORBIDDEN_FUNCS, self.FORBIDDEN_MODULES, self.ALLOWED_IMPORTS)
        checker.visit(self.tree)
        if checker.violations:
            return False, "Security Violation: " + ", ".join(checker.violations)
        
        # 3. Execution & Signature Check
        try:
            # Create a restricted execution environment
            safe_globals = {
                'np': np,
                'numpy': np,
                'math': __import__('math'),
                '__builtins__': {k: v for k, v in __builtins__.items() if k not in self.FORBIDDEN_FUNCS}
            }
            local_namespace = {}
            
            # Execute the code to define the function in memory
            exec(self.code, safe_globals, local_namespace)
            
            if 'calculate_reward' not in local_namespace:
                return False, "Function 'calculate_reward' not found. You must name the function exactly 'calculate_reward'."
            
            calc_func = local_namespace['calculate_reward']
            
            # Create Dummy Inputs matching LunarLander-v3 dimensions
            dummy_obs = np.zeros(8, dtype=np.float32)
            dummy_info = {
                'prev_obs': np.zeros(8, dtype=np.float32),
                'action': 0,
                'current_step': 10
            }
            
            # Test Execution
            result = calc_func(dummy_obs, dummy_info)
            
            # Strict Signature Validation
            if not isinstance(result, tuple) or len(result) != 2:
                return False, f"Function must return a Tuple of length 2 (total_reward, components_dict), got {type(result)}."
            
            total_reward, components = result
            
            # Check the Total Reward type (allow standard python floats or numpy floats)
            if not isinstance(total_reward, (float, int, np.floating, np.integer)):
                return False, f"First return value must be a numeric float, got {type(total_reward)}."
            
            # Check the Components Dictionary type
            if not isinstance(components, dict):
                return False, f"Second return value must be a dictionary, got {type(components)}."
            if len(components) == 0:
                return False, "The components dictionary is empty. It must contain the granular mathematical terms."
                
            for k, v in components.items():
                if not isinstance(k, str):
                    return False, f"Component dict keys must be strings, got {type(k)} for key {k}."
                if not isinstance(v, (float, int, np.floating, np.integer)):
                    return False, f"Component dict values must be numeric, got {type(v)} for key '{k}'."
                    
        except Exception as e:
            # Catch mathematical errors (e.g., dividing by zero), undefined variables, or attribute errors
            return False, f"Runtime Execution Error: {type(e).__name__}: {str(e)}"
            
        # If it passes syntax, security, execution, and signature checking, it is safe to write to disk.
        return True, self.code

# =========================================================================
# AST UTILITIES
# =========================================================================
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

    def visit_ImportFrom(self, node):
        if node.module:
            base = node.module.split('.')[0]
            if base in self.forbidden_modules or base not in self.allowed_imports:
                self.violations.append(f"from '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.forbidden_funcs:
            self.violations.append(f"function '{node.func.id}()'")
        self.generic_visit(node)