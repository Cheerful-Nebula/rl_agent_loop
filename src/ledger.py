import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

class ExperimentLedger:
    def __init__(self, workspace_root: Path):
        """
        Initialize the Ledger within the workspace structure.
        """
        self.filepath = workspace_root / "cognition" / "experiment_ledger.json"
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️  Ledger corrupted. Starting fresh.")
        
        return {"project": "RL_Agent_Loop", "experiments": []}

    def _save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=4)

    def get_last_experiment(self) -> Optional[Dict]:
        """Returns the most recent experiment (useful for Phase 0 validation)."""
        if not self.history["experiments"]:
            return None
        return self.history["experiments"][-1]

    def start_experiment(self, iteration: int, hypothesis: str, config_changes: Dict[str, Any]):
        """Phase 4: Open a new chapter."""
        new_entry = {
            "id": iteration,
            "timestamp": datetime.now().isoformat(),
            "status": "RUNNING",
            "hypothesis": hypothesis,
            "config_changes": config_changes,
            "validation": None, # Will be filled by Phase 0 of next run
            "metrics": {}
        }
        
        # Upsert logic (in case of re-runs)
        existing = next((e for e in self.history["experiments"] if e["id"] == iteration), None)
        if existing:
            index = self.history["experiments"].index(existing)
            self.history["experiments"][index] = new_entry
        else:
            self.history["experiments"].append(new_entry)
            
        self.history["experiments"].sort(key=lambda x: x["id"])
        self._save()
        print(f"📒 Ledger: Experiment {iteration} logged as RUNNING.")

    def update_validation(self, iteration: int, metrics: Dict, validation_result: Dict):
        """Phase 0: Close the previous chapter with Validation Data."""
        experiment = next((e for e in self.history["experiments"] if e["id"] == iteration), None)
        
        if experiment:
            # We save the high-level decision, not just raw metrics
            experiment["metrics"] = self._sanitize_metrics(metrics)
            experiment["validation"] = validation_result # {"is_validated": True, "reason": "..."}
            experiment["status"] = "COMPLETED"
            
            # Outcome string for the table
            if validation_result.get("is_validated"):
                experiment["outcome"] = "VALIDATED"
            else:
                experiment["outcome"] = "REFUTED"
                
            self._save()
            print(f"📒 Ledger: Experiment {iteration} closed as {experiment['outcome']}.")

    def _sanitize_metrics(self, metrics: Dict) -> Dict:
        """Extracts only the critical signal for the table."""
        # Note: You can customize this based on your 'performance' dict structure
        return {
            "Rew": round(metrics.get("mean_reward", -999), 1),
            "Success": round(metrics.get("reward_success_rate", 0), 2),
            "Crash": round(metrics.get("crash_rate", 0), 2),
            "Tilt": round(metrics.get("avg_tilt_angle", 0), 3)
        }

    def get_context_for_llm(self, limit: int = 5) -> str:
        """
        Returns the Markdown Table for the Analyst.
        """
        completed = [e for e in self.history["experiments"] if e.get("status") == "COMPLETED"]
        recent = completed[-limit:]
        
        if not recent:
            return "No verified history available yet."

        headers = "| Iter | Hypothesis | Changes | Metrics | Validation Result |"
        divider = "| :--- | :--- | :--- | :--- | :--- |"
        rows = []
        
        for exp in recent:
            changes = str(exp.get("config_changes", "{}"))[:50]
            metrics = " ".join([f"{k}:{v}" for k,v in exp["metrics"].items()])
            
            # The Critical Column: Did the prediction work?
            val = exp.get("validation", {})
            outcome_str = "✅ VALIDATED" if val.get("is_validated") else "❌ REFUTED"
            reason_str = val.get("reasoning", "")[:60] # Truncate for token limit
            
            row = f"| {exp['id']} | {exp['hypothesis'][:60]} | {changes} | {metrics} | {outcome_str}: {reason_str} |"
            rows.append(row)
            
        return "\n".join([headers, divider] + rows)