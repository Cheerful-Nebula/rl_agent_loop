import os
import json
from pathlib import Path
from datetime import datetime

class ExperimentWorkspace:
    def __init__(self, base_dir="experiments"):
        """
        Initializes the workspace by detecting the Campaign and Model 
        from Environment Variables set by outer_loop.sh.
        """
        # 1. CAPTURE CONTEXT FROM BASH
        # The outer_loop.sh exports CAMPAIGN_TAG and LLM_MODEL.
        # We catch them here to build our path.
        self.campaign_tag = os.environ.get("CAMPAIGN_TAG")
        self.raw_model_name = os.environ.get("LLM_MODEL")

        # Fallback for manual debugging (if running python without bash)
        if not self.campaign_tag:
            print("⚠️  No Campaign Tag found in env. Using 'Manual_Debug_Run'")
            self.campaign_tag = f"Manual_Debug_{datetime.now().strftime('%Y-%m-%d')}"
        
        if not self.raw_model_name:
            print("⚠️  No Model Name found in env. Using 'Debug_Model'")
            self.raw_model_name = "Debug_Model"

        # 2. SANITIZE MODEL NAME
        # Filesystems hate colons (llama3.1:8b -> llama3.1-8b)
        self.model_dir_name = self.raw_model_name.replace(":", "-")

        # 3. CONSTRUCT THE HIERARCHY
        # Structure: experiments / {Campaign_Tag} / {Model_Name} / {Category}
        self.campaign_path = Path(base_dir) / self.campaign_tag
        self.model_root_path = self.campaign_path / self.model_dir_name
        
        # 4. DEFINE SUBDIRECTORIES (The Standard Structure)
        self.dirs = {
            "root": self.model_root_path,
            "cognition": self.model_root_path / "cognition",
            "code": self.model_root_path / "generated_code",
            "tensorboard": self.model_root_path / "telemetry" / "tensorboard",
            "telemetry_raw": self.model_root_path / "telemetry" / "raw",
            "plots": self.model_root_path / "artifacts" / "plots",
            "models": self.model_root_path / "artifacts" / "models",
            "videos": self.model_root_path / "artifacts" / "videos"
        }

        # 5. BUILD IT
        self._create_directories()
        
        # Only print this once per run to keep logs clean
        # (You can suppress this if it gets too noisy in the loop)
        # print(f"📍 Workspace Active: {self.model_root_path}")

    def _create_directories(self):
        """Recursively creates the directory tree."""
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_path(self, category, iteration, filename):
        """
        Returns a sorted, standardized path for a file.
        Example: .../llama3.1-8b/cognition/iter05_reasoning.md
        """
        if category not in self.dirs:
            raise ValueError(f"Category '{category}' not defined in workspace.")

        # Naming convention: iterXX_filename
        clean_filename = f"iter{int(iteration):02d}_{filename}"
        return self.dirs[category] / clean_filename

    # --- DATA HANDOFF HELPERS ---

    def save_metrics(self, iteration, metrics_dict):
        """Saves a JSON report card for a specific iteration."""
        filepath = self.get_path("telemetry_raw", iteration, "metrics.json")
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=4)

    def load_metrics(self, iteration):
        """Loads the JSON report card from a specific iteration."""
        filepath = self.get_path("telemetry_raw", iteration, "metrics.json")
        if not filepath.exists():
            return None
        
        with open(filepath, "r") as f:
            return json.load(f)