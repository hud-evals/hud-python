#!/usr/bin/env python3
"""
Test script to validate all inspect_evals with our framework.

This script iterates through all evals in available_evals.txt and tests
whether they can be successfully converted to Hud task format.
"""

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx


def read_eval_list(file_path: str = "available_evals.txt") -> list[str]:
    """Read list of eval names from file."""
    with open(file_path) as f:
        evals = [
            line.strip() for line in f if line.strip() and not line.startswith("=")
        ]
    return evals


def read_confirmed_working(file_path: str) -> set[str]:
    """Read list of confirmed working eval names from file."""
    if not Path(file_path).exists():
        return set()
    with open(file_path) as f:
        return {line.strip() for line in f if line.strip()}


def append_confirmed_working(eval_name: str, file_path: str) -> None:
    """Append an eval name to the confirmed working file."""
    with open(file_path, "a") as f:
        f.write(f"{eval_name}\n")
    print(f"  💾 Saved to {file_path}")


def check_mcp_server(url: str = "http://localhost:8765/mcp", timeout: float = 2.0) -> bool:
    """
    Check if MCP server is reachable.

    Args:
        url: MCP server URL
        timeout: Timeout in seconds

    Returns:
        True if server is reachable, False otherwise
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            # Try to connect to the server
            response = client.get(url, follow_redirects=True)
            return response.status_code < 500
    except Exception:
        return False


def test_eval(eval_name: str, test_execution: bool = True, timeout: int = 300) -> dict:
    """
    Test a single eval by running prepare_dataset.py with limit=1.
    Optionally also test running the actual eval with hud.

    Args:
        eval_name: Name of the eval to test
        test_execution: If True, also run 'hud eval samples.jsonl' after preparation
        timeout: Timeout in seconds for prepare_dataset

    Returns:
        Dict with 'eval', 'status', 'output', 'error' keys
    """
    print(f"  Testing {eval_name}...", end=" ", flush=True)

    # Clean up any existing samples.jsonl
    samples_file = Path("samples.jsonl")
    if samples_file.exists():
        samples_file.unlink()

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "prepare_dataset.py",
                "--eval",
                eval_name,
                "--limit",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check if samples.jsonl was created and is valid
        if not samples_file.exists():
            print("❌ FAIL (no output file)")
            return {
                "eval": eval_name,
                "status": "FAIL",
                "prep_status": "FAIL",
                "exec_status": None,
                "output": result.stdout[-500:],
                "error": f"No samples.jsonl created. stderr: {result.stderr[-200:]}",
            }

        try:
            with open(samples_file) as f:
                task = json.loads(f.readline())
                # Verify it has expected fields
                if not ("id" in task and "prompt" in task and "agent_tools" in task):
                    print("❌ FAIL (invalid task format)")
                    return {
                        "eval": eval_name,
                        "status": "FAIL",
                        "prep_status": "FAIL",
                        "exec_status": None,
                        "output": result.stdout[-500:],
                        "error": "Task missing required fields",
                    }
        except json.JSONDecodeError as e:
            print("❌ FAIL (invalid JSON)")
            return {
                "eval": eval_name,
                "status": "FAIL",
                "prep_status": "FAIL",
                "exec_status": None,
                "output": result.stdout[-500:],
                "error": f"JSON decode error: {e}",
            }

        # Phase 1 (preparation) passed
        tools = task.get("agent_tools", [])
        prep_output = (
            result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
        )

        # Phase 2: Execute eval if requested
        if test_execution:
            print("✅ PREP", end=" ", flush=True)
            print("→ EXEC...", end=" ", flush=True)

            try:
                exec_result = subprocess.run(
                    ["hud", "eval", "samples.jsonl", "claude"],
                    capture_output=True,
                    text=True,
                    timeout=timeout * 2,  # Give more time for execution
                )

                # Check if execution succeeded
                exec_output = exec_result.stdout + exec_result.stderr
                if exec_result.returncode == 0:
                    print("✅ EXEC")
                    return {
                        "eval": eval_name,
                        "status": "PASS",
                        "prep_status": "PASS",
                        "exec_status": "PASS",
                        "output": prep_output,
                        "exec_output": (
                            exec_output[-500:]
                            if len(exec_output) > 500
                            else exec_output
                        ),
                        "error": None,
                        "tools": tools,
                    }
                else:
                    print("❌ EXEC FAIL")
                    return {
                        "eval": eval_name,
                        "status": "EXEC_FAIL",
                        "prep_status": "PASS",
                        "exec_status": "FAIL",
                        "output": prep_output,
                        "exec_output": (
                            exec_output[-500:]
                            if len(exec_output) > 500
                            else exec_output
                        ),
                        "error": f"Execution failed with return code {exec_result.returncode}",
                        "tools": tools,
                    }

            except subprocess.TimeoutExpired:
                print("⏱️  EXEC TIMEOUT")
                return {
                    "eval": eval_name,
                    "status": "EXEC_TIMEOUT",
                    "prep_status": "PASS",
                    "exec_status": "TIMEOUT",
                    "output": prep_output,
                    "exec_output": "",
                    "error": f"Execution timed out after {timeout * 2}s",
                    "tools": tools,
                }
            except Exception as e:
                print(f"❌ EXEC ERROR")
                return {
                    "eval": eval_name,
                    "status": "EXEC_ERROR",
                    "prep_status": "PASS",
                    "exec_status": "ERROR",
                    "output": prep_output,
                    "exec_output": "",
                    "error": f"Execution error: {str(e)}",
                    "tools": tools,
                }
        else:
            # Only tested preparation
            print("✅ PASS")
            return {
                "eval": eval_name,
                "status": "PASS",
                "prep_status": "PASS",
                "exec_status": None,
                "output": prep_output,
                "error": None,
                "tools": tools,
            }

    except subprocess.TimeoutExpired:
        print("⏱️  TIMEOUT")
        return {
            "eval": eval_name,
            "status": "TIMEOUT",
            "prep_status": "TIMEOUT",
            "exec_status": None,
            "output": "",
            "error": f"Timed out after {timeout}s",
        }
    except Exception as e:
        print(f"❌ ERROR")
        return {
            "eval": eval_name,
            "status": "ERROR",
            "prep_status": "ERROR",
            "exec_status": None,
            "output": "",
            "error": str(e),
        }
    finally:
        # Clean up samples file
        if samples_file.exists():
            samples_file.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Test all inspect_evals with the Hud framework"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of evals to test (for quick testing)",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip execution testing (only test dataset preparation)",
    )
    parser.add_argument(
        "--confirmed-working",
        type=str,
        default="confirmed_working.txt",
        help="File containing confirmed working evals to skip (default: confirmed_working.txt)",
    )
    args = parser.parse_args()

    print("🧪 Testing inspect_evals with our framework\n")
    print("=" * 70)

    test_execution = not args.skip_execution

    # Check if MCP server is running (needed for execution)
    if test_execution:
        print("Checking MCP server availability...", end=" ", flush=True)
        if check_mcp_server():
            print("✅ MCP server is running\n")
        else:
            print("❌ Not running\n")
            print("❌ MCP server not reachable at http://localhost:8765/mcp")
            print("   Run `hud dev --build` first to start the sandbox server")
            print("\n   Or use --skip-execution to only test dataset preparation")
            sys.exit(1)
    else:
        print("⚠️  Execution testing skipped - only testing dataset preparation\n")

    # Read eval list
    try:
        eval_list = read_eval_list()
    except FileNotFoundError:
        print("❌ available_evals.txt not found. Run list_all_evals.py first.")
        sys.exit(1)

    # Load confirmed working evals to skip
    confirmed_working = read_confirmed_working(args.confirmed_working)
    if confirmed_working:
        print(f"📋 Loaded {len(confirmed_working)} confirmed working evals from {args.confirmed_working}")
        # Filter out confirmed working evals
        original_count = len(eval_list)
        eval_list = [e for e in eval_list if e not in confirmed_working]
        skipped_count = original_count - len(eval_list)
        if skipped_count > 0:
            print(f"⏩ Skipping {skipped_count} already confirmed working evals\n")
    else:
        print(f"📋 No confirmed working file found at {args.confirmed_working}\n")

    # Apply limit if specified (random sample)
    if args.limit:
        if args.limit < len(eval_list):
            eval_list = random.sample(eval_list, args.limit)
            print(f"Testing random sample of {len(eval_list)} evals\n")
            print(f"Selected: {', '.join(eval_list)}\n")
        else:
            print(
                f"Limit ({args.limit}) >= total evals ({len(eval_list)}), testing all\n"
            )
    else:
        print(f"Found {len(eval_list)} evals to test\n")

    # Test each eval
    results = []
    start_time = datetime.now()
    output_file = "eval_test_results.json"

    for i, eval_name in enumerate(eval_list, 1):
        print(f"[{i}/{len(eval_list)}]", end=" ")
        result = test_eval(eval_name, test_execution=test_execution)
        results.append(result)

        # If eval passed both prep and exec, immediately save to confirmed_working
        if (
            result["status"] == "PASS"
            and result.get("prep_status") == "PASS"
            and (not test_execution or result.get("exec_status") == "PASS")
        ):
            append_confirmed_working(eval_name, args.confirmed_working)

        # Save results incrementally after each eval
        with open(output_file, "w") as f:
            json.dump(
                {
                    "timestamp": start_time.isoformat(),
                    "duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "total": len(results),
                    "completed": len(results),
                    "remaining": len(eval_list) - len(results),
                    "results": results,
                },
                f,
                indent=2,
            )

    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Overall stats
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] in ["FAIL", "EXEC_FAIL"])
    timeout = sum(1 for r in results if r["status"] in ["TIMEOUT", "EXEC_TIMEOUT"])
    errors = sum(1 for r in results if r["status"] in ["ERROR", "EXEC_ERROR"])

    # Preparation phase stats
    prep_passed = sum(1 for r in results if r.get("prep_status") == "PASS")
    prep_failed = sum(1 for r in results if r.get("prep_status") == "FAIL")

    # Execution phase stats (only if execution testing was enabled)
    if test_execution:
        exec_passed = sum(1 for r in results if r.get("exec_status") == "PASS")
        exec_failed = sum(1 for r in results if r.get("exec_status") == "FAIL")
        exec_timeout = sum(1 for r in results if r.get("exec_status") == "TIMEOUT")
        exec_error = sum(1 for r in results if r.get("exec_status") == "ERROR")

    # Save final detailed results with statistics
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "total": len(results),
                "completed": len(results),
                "passed": passed,
                "failed": failed,
                "timeout": timeout,
                "errors": errors,
                "results": results,
            },
            f,
            indent=2,
        )

    # Create summary report
    summary_file = "eval_test_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Inspect Evals Framework Test Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {start_time}\n")
        f.write(f"Duration: {duration:.1f}s\n")
        f.write(f"Total Evals Tested: {len(results)}")
        if args.limit and args.limit < len(read_eval_list()):
            f.write(f" (random sample of {args.limit})")
        f.write("\n")
        f.write(f"Execution Testing: {'Enabled' if test_execution else 'Disabled'}\n")
        f.write("\n")

        # Overall results
        f.write("OVERALL RESULTS:\n")
        f.write(f"✅ Passed:  {passed:3d} ({passed/len(results)*100:.1f}%)\n")
        f.write(f"❌ Failed:  {failed:3d} ({failed/len(results)*100:.1f}%)\n")
        f.write(f"⏱️  Timeout: {timeout:3d} ({timeout/len(results)*100:.1f}%)\n")
        f.write(f"💥 Errors:  {errors:3d} ({errors/len(results)*100:.1f}%)\n")
        f.write("\n")

        # Phase-specific stats
        f.write("PREPARATION PHASE:\n")
        f.write(f"✅ Passed:  {prep_passed:3d} ({prep_passed/len(results)*100:.1f}%)\n")
        f.write(f"❌ Failed:  {prep_failed:3d} ({prep_failed/len(results)*100:.1f}%)\n")
        f.write("\n")

        if test_execution:
            f.write("EXECUTION PHASE:\n")
            if prep_passed > 0:
                f.write(
                    f"✅ Passed:  {exec_passed:3d} ({exec_passed/prep_passed*100:.1f}% of prepared)\n"
                )
                f.write(
                    f"❌ Failed:  {exec_failed:3d} ({exec_failed/prep_passed*100:.1f}% of prepared)\n"
                )
                f.write(
                    f"⏱️  Timeout: {exec_timeout:3d} ({exec_timeout/prep_passed*100:.1f}% of prepared)\n"
                )
                f.write(
                    f"💥 Errors:  {exec_error:3d} ({exec_error/prep_passed*100:.1f}% of prepared)\n"
                )
            else:
                f.write("  (no successful preparations to execute)\n")
            f.write("\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("PASSED EVALS:\n")
        f.write("=" * 70 + "\n")
        for r in results:
            if r["status"] == "PASS":
                tools_str = ", ".join(r.get("tools", []))
                f.write(f"✅ {r['eval']:<30} [{tools_str}]\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("FAILED EVALS:\n")
        f.write("=" * 70 + "\n")
        for r in results:
            if r["status"] in ["FAIL", "TIMEOUT", "ERROR"]:
                f.write(f"{r['status']:8s} {r['eval']:<30}\n")
                if r["error"]:
                    error_preview = r["error"][:100]
                    if len(r["error"]) > 100:
                        error_preview += "..."
                    f.write(f"         {error_preview}\n")
                f.write("\n")

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total:    {len(results)}")
    print(f"\nOVERALL:")
    print(f"✅ Passed:  {passed:3d} ({passed/len(results)*100:.1f}%)")
    print(f"❌ Failed:  {failed:3d} ({failed/len(results)*100:.1f}%)")
    print(f"⏱️  Timeout: {timeout:3d} ({timeout/len(results)*100:.1f}%)")
    print(f"💥 Errors:  {errors:3d} ({errors/len(results)*100:.1f}%)")

    print(f"\nPREPARATION PHASE:")
    print(f"✅ Passed:  {prep_passed:3d} ({prep_passed/len(results)*100:.1f}%)")
    print(f"❌ Failed:  {prep_failed:3d} ({prep_failed/len(results)*100:.1f}%)")

    if test_execution:
        print(f"\nEXECUTION PHASE:")
        if prep_passed > 0:
            print(
                f"✅ Passed:  {exec_passed:3d} ({exec_passed/prep_passed*100:.1f}% of prepared)"
            )
            print(
                f"❌ Failed:  {exec_failed:3d} ({exec_failed/prep_passed*100:.1f}% of prepared)"
            )
            print(
                f"⏱️  Timeout: {exec_timeout:3d} ({exec_timeout/prep_passed*100:.1f}% of prepared)"
            )
            print(
                f"💥 Errors:  {exec_error:3d} ({exec_error/prep_passed*100:.1f}% of prepared)"
            )
        else:
            print("  (no successful preparations to execute)")

    print(f"\nDuration: {duration:.1f}s")
    print(f"\n📊 Detailed results: {output_file}")
    print(f"📝 Summary report: {summary_file}")


if __name__ == "__main__":
    main()
