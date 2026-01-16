import ast
import numpy as np
import importlib.util
from typing import Tuple, Optional, Dict, Any

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
        2. Integrity: Check for Crashes, NaNs, Infinities, and Type errors.
        3. Logic: Check for Vanishing Gradients, Exploding Rewards, and Incentive Alignment.
        Args:
            strict_mode (bool): If False, skips the 'Logic Checks' (Vanishing Gradient/Incentives).
                                Useful for Iteration 1 to allow imperfect heuristics to pass.
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
                # In non-strict mode, we still run the code to ensure it DOESN'T CRASH 
                # on the logic scenarios, but we ignore the boolean result.
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

    def _get_mock_data(self, scenario_type="standard"):
        """
        Generates Mock Data that mimics your Wrapper's 'build_physical_state'.
        We manually populate 'distance_from_origin' etc. to match the physics of the scenario.
        """
        # Base keys that your Wrapper provides directly from Box2D
        data = {
            "x_pos": 0.0, "y_pos": 0.0,
            "x_vel": 0.0, "y_vel": 0.0,
            "angle": 0.0, "angular_velocity": 0.0,
            "distance_from_origin": 0.0,    # Wrapper pulls this from lander.position.length
            "linear_velocity_mag": 0.0,     # Wrapper pulls this from lander.linearVelocity.length
            "fuel_consumed_this_step": 0.0
        }

        if scenario_type == "hover_high":
            # Far away, stationary
            data.update({
                "y_pos": 1.0, 
                "distance_from_origin": 1.0, 
                "linear_velocity_mag": 0.0
            })
        
        elif scenario_type == "landing_pad":
            # Perfect spot, gentle speed
            data.update({
                "y_pos": 0.0, 
                "distance_from_origin": 0.0, 
                "linear_velocity_mag": 0.1, 
                "y_vel": -0.1
            })
            
        elif scenario_type == "crash_fast":
            # Perfect spot, but hitting hard
            data.update({
                "y_pos": 0.0,
                "distance_from_origin": 0.0, 
                "linear_velocity_mag": 5.0,
                "y_vel": -5.0
            })

        return data

    def _check_math_integrity(self, module) -> Tuple[bool, str]:
        """
        Phase 1: Integrity Check.
        Tests for Runtime Crashes, NaNs, Infinities, and Return Types.
        """
        # Test Case 1: Division by Zero Risk (Zero Distance)
        zero_data = self._get_mock_data("landing_pad")
        zero_data["distance_from_origin"] = 0.0 # Force zero
        zero_data["linear_velocity_mag"] = 0.0
        
        # Test Case 2: Standard Values (Sanity Check)
        std_data = self._get_mock_data("hover_high")

        # PPO Agent sees zeros (LLM should ignore this)
        dummy_obs = [0.0] * 8

        for name, raw_physics in [("Zero-Input (Edge Case)", zero_data), ("Standard-Input", std_data)]:
            try:
                # Construct the info dict exactly like the Wrapper does
                info = {"raw_physics": raw_physics}
                
                reward = module.calculate_reward(dummy_obs, info)

                # Check A: Return Type
                if not isinstance(reward, (float, np.floating, int, np.integer)):
                    return False, f"Type Error: Function returned {type(reward)}, expected float."

                # Check B: Math Integrity
                if np.isnan(reward):
                    return False, f"Math Error: Reward is NaN in {name}. Check for 0/0 or log(-1)."
                
                if np.isinf(reward):
                    return False, f"Math Error: Reward is Infinite in {name}. Check for division by zero."

            except Exception as e:
                return False, f"Crash in {name}: {str(e)}"

        return True, ""

    def _check_rl_logic(self, module) -> Tuple[bool, str]:
        """
        Phase 2: Logic Check (The Dual View).
        Tests for Vanishing Gradients, Exploding Gradients, and Inverse Rewards.
        """
        scenarios = ["hover_high", "landing_pad", "crash_fast"]
        results = {}
        dummy_obs = [0.0] * 8 

        # 1. Collect Rewards
        for name in scenarios:
            raw_physics = self._get_mock_data(name)
            info = {"raw_physics": raw_physics}
            try:
                reward = module.calculate_reward(dummy_obs, info)
                results[name] = float(reward)
            except Exception as e:
                return False, f"Logic Check Failed: Crash in scenario '{name}' ({str(e)})"

        # 2. Check: Exploding Gradients (Normalization)
        for name, r in results.items():
            if abs(r) > 20.0: # Loose bound
                return False, f"Scaling Error: Reward in '{name}' is {r:.2f}. Exploding Gradient Risk. Normalize to approx -1.0 to 1.0."

        # 3. Check: Vanishing Gradient (Incentive to Land)
        # Landing (0.0 distance) must be better than Hovering (1.0 distance)
        if results["landing_pad"] <= results["hover_high"]:
            return False, (
                f"Logic Error (Vanishing Gradient): Landing ({results['landing_pad']:.3f}) "
                f"is not better than Hovering ({results['hover_high']:.3f}). "
                "The agent has no incentive to go down."
            )

        # 4. Check: Safety (Incentive not to Crash)
        # Soft landing must be better than hard crash
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