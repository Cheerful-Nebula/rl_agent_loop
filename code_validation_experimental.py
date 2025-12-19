import ast
import tempfile
import os
import logging
from vulture import Vulture
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RewardValidator")

class RewardValidator:
    def __init__(self):
        # 1. WHITELIST: The only imports strictly allowed for a reward function
        self.ALLOWED_IMPORTS = {'math', 'numpy', 'np', 'gym', 'gymnasium', 'random'}
        
        # 2. BLACKLIST: Explicit dangerous functions we want to catch immediately
        self.DANGEROUS_FUNCTIONS = {'eval', 'exec', 'compile', 'open', 'input', 'system'}
        
        # 3. DANGEROUS MODULES: Modules that signal the Agent is trying to do too much
        self.DANGEROUS_MODULES = {'os', 'sys', 'subprocess', 'shutil', 'requests', 'socket'}

    def validate_code(self, code_str: str) -> Dict[str, Any]:
        """
        Main entry point. Runs all checks on the provided code string.
        Returns a dict with 'valid' (bool) and 'feedback' (str).
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "feedback": ""
        }

        # --- STEP 1: Syntax Check (AST) ---
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            results["valid"] = False
            results["errors"].append(f"Syntax Error on line {e.lineno}: {e.msg}")
            results["feedback"] = f"The code generated has a Syntax Error: {e.msg}. Please rewrite it."
            return results

        # --- STEP 2: Security & Import Check (AST Visitor) ---
        security_visitor = self._SecurityVisitor(self.ALLOWED_IMPORTS, self.DANGEROUS_FUNCTIONS, self.DANGEROUS_MODULES)
        security_visitor.visit(tree)
        
        if security_visitor.errors:
            results["valid"] = False
            results["errors"].extend(security_visitor.errors)
            # Construct the "Whitelisted Method" prompt
            results["feedback"] = self._generate_whitelist_feedback(security_visitor.errors)
            return results

        # --- STEP 3: Dead Code & Logic Check (Vulture) ---
        # Vulture works best on files, so we create a temp file
        vulture_issues = self._run_vulture_check(code_str)
        if vulture_issues:
            # We treat dead code as a warning, not a hard failure, 
            # unless it suggests logic errors (like unused variables in calculation).
            results["warnings"].extend(vulture_issues)
            results["feedback"] += "\nNote: " + "; ".join(vulture_issues)

        return results

    def _generate_whitelist_feedback(self, errors: List[str]) -> str:
        """Generates the prompt string to guide the LLM back to safety."""
        return (
            f"Security Check Failed: {'; '.join(errors)}. "
            f"You are restricted to the following safe imports: {list(self.ALLOWED_IMPORTS)}. "
            "Please rewrite the code using ONLY standard math or numpy operations. "
            "Do not access the file system, network, or system internals."
        )

    def _run_vulture_check(self, code_str: str) -> List[str]:
        """Runs Vulture on a temporary file to find dead code."""
        issues = []
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code_str)
            tmp_path = tmp.name

        try:
            v = Vulture(verbose=False)
            v.scan(tmp_path)
            # Filter results: Vulture often flags the main function definition as unused
            # because nothing calls it in the script itself. We ignore that specific case.
            for item in v.get_unused_code(min_confidence=60):
                if item.typ == 'def' and item.name == 'calculate_reward':
                    continue # This is expected
                issues.append(f"Unused {item.typ} '{item.name}' found (Dead Code)")
        except Exception as e:
            logger.warning(f"Vulture check failed: {e}")
        finally:
            os.remove(tmp_path)
            
        return issues

    # --- Internal AST Visitor Class ---
    class _SecurityVisitor(ast.NodeVisitor):
        def __init__(self, allowed_imports, dangerous_funcs, dangerous_modules):
            self.errors = []
            self.allowed_imports = allowed_imports
            self.dangerous_funcs = dangerous_funcs
            self.dangerous_modules = dangerous_modules

        def visit_Import(self, node):
            for alias in node.names:
                self._check_import(alias.name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module:
                self._check_import(node.module)
            self.generic_visit(node)

        def visit_Call(self, node):
            # Check for calling dangerous functions like eval() or exec()
            if isinstance(node.func, ast.Name):
                if node.func.id in self.dangerous_funcs:
                    self.errors.append(f"Use of dangerous function '{node.func.id}' is prohibited")
            self.generic_visit(node)

        def _check_import(self, module_name):
            base_module = module_name.split('.')[0]
            if base_module in self.dangerous_modules:
                self.errors.append(f"Importing '{base_module}' is strictly forbidden")
            elif base_module not in self.allowed_imports:
                self.errors.append(f"Import '{base_module}' is not in the allowed whitelist")