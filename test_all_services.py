import subprocess
import sys
import os

SERVICES = [
    "asr_service",
    "logic_service",
    "phone_gateway/backend",
    "vad_service",
    "tts_service",
    "rag_service"
]

def run_tests():
    print("="*60)
    print("      Voice-To-Voice Farmer Advisor - Test Runner")
    print("="*60)
    
    all_passed = True
    results = []

    for service in SERVICES:
        print(f"\n>>> Testing {service}...")
        # We run each service in a separate process to avoid module naming collisions
        # (e.g., multiple 'main.py' or 'audio_utils.py' files).
        cmd = [sys.executable, "-m", "pytest", f"{service}/tests", "-v"]
        
        try:
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode == 0:
                results.append((service, "PASSED"))
            else:
                results.append((service, "FAILED"))
                all_passed = False
        except Exception as e:
            print(f"Error running tests for {service}: {e}")
            results.append((service, "ERROR"))
            all_passed = False

    print("\n" + "="*60)
    print("                FINAL TEST SUMMARY")
    print("="*60)
    for service, status in results:
        print(f"{service:<30} : {status}")
    print("="*60)
    
    if all_passed:
        print("All test suites passed successfully!")
        sys.exit(0)
    else:
        print("Some test suites failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()