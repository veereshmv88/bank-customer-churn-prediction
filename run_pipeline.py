"""
Complete End-to-End Orchestrator Pipeline.
Runs data loading, data audit, model training, evaluation, report generation, and verification.
"""
import sys
import subprocess
import time

def run_step(description: str, command: list):
    print(f"\n>>> [PIPELINE STEP] {description}...")
    start = time.time()
    result = subprocess.run(command, text=True, capture_output=True)
    duration = time.time() - start
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error in {description}:")
        print(result.stderr)
        sys.exit(result.returncode)
    print(f">>> Completed in {duration:.2f}s")

def main():
    print("=================================================================")
    print("BANK CUSTOMER CHURN: COMPLETE REPRODUCIBLE PIPELINE")
    print("=================================================================")
    run_step("Train Models & Generate Artifacts", [sys.executable, "train.py"])
    run_step("Evaluate Model on Held-Out Test Set", [sys.executable, "evaluate.py"])
    run_step("Run Automated Pytest Test Suite", [sys.executable, "-m", "pytest", "tests/", "-v"])
    print("\n=================================================================")
    print("PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
    print("To launch the web application, run: python app.py")
    print("=================================================================")

if __name__ == "__main__":
    main()
