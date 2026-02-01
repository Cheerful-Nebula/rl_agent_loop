import ast
import numpy as np
import importlib.util
from typing import Tuple, Optional, Dict, Any, List

class CodeValidator:
    def __init__(self, code: str):
        self.code = code
        self.tree: Optional[ast.AST] = None
        self.error_message: Optional[str] = None
        
        # Security: Whitelist allowed libraries
        self.ALLOWED_IMPORTS = {'math', 'numpy', 'np', 'gym', 'gymnasium'}
        self.FORBIDDEN_FUNCS = {'eval', 'exec', 'open', 'input', 'system', 'subprocess'}
        self.FORBIDDEN_MODULES = {'os', 'sys', 'shutil', 'requests', 'socket'}

        try:
            self.tree = ast.parse(code)
        except SyntaxError as e:
            self.error_message = f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"

    # =========================================================================
    # 1. STATIC ANALYSIS (Security & Syntax)
    # =========================================================================
    def validate_static(self) -> Tuple[bool, str]:
        """Performs Static Analysis (Syntax + Security)."""
        if self.tree is None:
            return False, self.error_message

        checker = BlacklistChecker(self.FORBIDDEN_FUNCS, self.FORBIDDEN_MODULES, self.ALLOWED_IMPORTS)
        checker.visit(self.tree)
        
        if checker.violations:
            feedback = "Security Violation: " + ", ".join(checker.violations)
            return False, feedback

        return True, "Static checks passed."

    # =========================================================================
    # 2. RUNTIME ORCHESTRATION
    # =========================================================================
    def validate_runtime(self, strict_mode: bool = True) -> Tuple[bool, str]:
        """
        Orchestrates the validation pipeline:
        1. Compile: Load code into a sandbox (Holodeck).
        2. Integrity: Check for Crashes, NaNs, Infinities, and Type errors with new signature.
        3. Logic: Check for Vanishing Gradients, Exploding Rewards, and Incentive Alignment.
        Args:
            strict_mode (bool): If False, skips the 'Logic Checks' (Vanishing Gradient/Incentives).
        """
        try:
            # A. Build the Sandbox
            module = self._create_holodeck()
            
            # B. Check Math Integrity (Computer Science Errors)
            valid_math, msg_math = self._check_math_integrity(module)
            if not valid_math:
                return False, msg_math

            # C. Check RL Logic (Reinforcement Learning Errors: Only run if strict_mode is True)
            if strict_mode:
                valid_logic, msg_logic = self._check_rl_logic(module)
                if not valid_logic:
                    return False, msg_logic
            else:
                self._check_rl_logic(module)

            return True, "Passed all Runtime Safety and Logic checks."

        except Exception as e:
            return False, f"Runtime Compilation Error: {str(e)}"

    # =========================================================================
    # 3. HELPER METHODS (The "Holodeck" Logic)
    # =========================================================================
    def _create_holodeck(self):
        """Creates a temporary, isolated module and executes the user code inside it."""
        spec = importlib.util.spec_from_loader("temp_reward_check", loader=None)
        module = importlib.util.module_from_spec(spec)
        # Execute the LLM's code inside this new, empty module object
        exec(self.code, module.__dict__)
        
        if not hasattr(module, 'calculate_reward'):
            raise ValueError("Function 'calculate_reward' not found in generated code.")
            
        return module

    def _get_mock_data(self, scenario_type="standard") -> Tuple[List[float], Dict[str, Any]]:
        """
        Generates Mock Data for the new function signature.
        
        Signature requires:
        1. observation (list): [x, y, vx, vy, ang, ang_v, leg1, leg2]
        2. info (dict): {"action_usage": {...}}
        """
        # Default: Hovering at (0,0) with no velocity
        # Obs: [x, y, vx, vy, ang, ang_v, leg1, leg2]
        obs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Default Info containing the required 'action_usage' dict
        info = {
            "action_usage": {
                "fuel_consumed_this_step": 0.0,
                "action_index": 0,
                "action_label": "nothing"
            }
        }

        if scenario_type == "hover_high":
            # High up (y=1.0), stationary, legs up
            obs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        elif scenario_type == "landing_pad":
            # On ground (y=0.0), legs touching (1.0), gentle downward velocity (-0.05)
            obs = [0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 1.0, 1.0]
            
        elif scenario_type == "crash_fast":
            # On ground (y=0.0), high downward velocity (-5.0), legs up
            obs = [0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0]
            
        elif scenario_type == "fuel_heavy_usage":
             # Same as hover, but firing main engine hard
             obs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
             info["action_usage"]["fuel_consumed_this_step"] = 10.0
             info["action_usage"]["action_index"] = 2

        return obs, info

    def _check_math_integrity(self, module) -> Tuple[bool, str]:
        """
        Phase 1: Integrity Check.
        Tests for Runtime Crashes, NaNs, Infinities, and Return Types using new signature.
        """
        # Test Case 1: Division by Zero Risk (Zero State)
        # We pass strictly zeros to check for 0/0 or log(0) errors on position/vel
        zero_obs = [0.0] * 8
        zero_info = {
            "action_usage": {
                "action_index": 0,
            }
        }
        
        # Test Case 2: Standard Values
        std_obs, std_info = self._get_mock_data("hover_high")

        # Test Case 3: Action Usage Edge Case (Ensures they handle the info dict)
        #fuel_obs, fuel_info = self._get_mock_data("fuel_heavy_usage")

        test_cases = [
            ("Zero-Input (Edge Case)", zero_obs, zero_info),
            ("Standard-Input", std_obs, std_info)
        #    ("Fuel-Usage-Input", fuel_obs, fuel_info)
        ]

        for name, obs, info in test_cases:
            try:
                reward = module.calculate_reward(obs, info)

                # Check A: Return Type
                if not isinstance(reward, (float, np.floating, int, np.integer)):
                    return False, f"Type Error: Function returned {type(reward)}, expected float."

                # Check B: Math Integrity
                if np.isnan(reward):
                    return False, f"Math Error: Reward is NaN in {name}. Check for 0/0 or log(-1)."
                
                if np.isinf(reward):
                    return False, f"Math Error: Reward is Infinite in {name}. Check for division by zero."

            except KeyError as e:
                return False, f"Key Error in {name}: You accessed a key that doesn't exist: {e}. Check docstring constraints."
            except IndexError as e:
                return False, f"Index Error in {name}: Accessed invalid index in observation list: {e}."
            except Exception as e:
                return False, f"Crash in {name}: {str(e)}"

        return True, ""

    def _check_rl_logic(self, module) -> Tuple[bool, str]:
        """
        Phase 2: Logic Check.
        Tests for Vanishing Gradients, Exploding Gradients, and Inverse Rewards.
        """
        scenarios = ["hover_high", "landing_pad", "crash_fast"]
        results = {}

        # 1. Collect Rewards
        for name in scenarios:
            obs, info = self._get_mock_data(name)
            try:
                reward = module.calculate_reward(obs, info)
                results[name] = float(reward)
            except Exception as e:
                return False, f"Logic Check Failed: Crash in scenario '{name}' ({str(e)})"

        # 2. Check: Exploding Gradients (Normalization)
        #for name, r in results.items():
        #    if abs(r) > 20.0: # Loose bound
        #        return False, f"Scaling Error: Reward in '{name}' is {r:.2f}. Exploding Gradient Risk. Normalize to approx -1.0 to 1.0."

        # 3. Check: Vanishing Gradient (Incentive to Land)
        # Landing (y=0) must be better than Hovering (y=1)
        if results["landing_pad"] <= results["hover_high"]:
            return False, (
                f"Logic Error (Vanishing Gradient): Landing ({results['landing_pad']:.3f}) "
                f"is not better than Hovering ({results['hover_high']:.3f}). "
                "The agent has no incentive to go down."
            )

        # 4. Check: Safety (Incentive not to Crash)
        # Soft landing (-0.05 vel) must be better than hard crash (-5.0 vel)
        if results["landing_pad"] <= results["crash_fast"]:
             return False, (
                f"Safety Error: Crashing ({results['crash_fast']:.3f}) "
                f"is rewarded better/equal to Landing ({results['landing_pad']:.3f}). "
                "Penalize velocity more."
            )

        return True, ""


# =========================================================================
# 4. AST UTILITIES
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