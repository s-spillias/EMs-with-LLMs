import os
import subprocess
import json
import pandas as pd
import time
import sys

# Use a safe, non-interactive backend in multi-process runs
import matplotlib
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

from scripts.model_report_handler import update_model_report, read_model_report


class TextColors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'


def log_message(message, color=TextColors.RESET):
    # Flush so logs are visible even when run inside parallel pools
    print(color + message + TextColors.RESET, flush=True)


def remove_error_files(individual_dir):
    error_file_path = os.path.join(individual_dir, 'model_error.log')
    if os.path.exists(error_file_path):
        os.remove(error_file_path)
        log_message(f"Removed error file: {error_file_path}", TextColors.YELLOW)


def plot_model_vs_historical(individual_dir, plot_data):
    """
    Delay pyplot import until here to avoid race conditions during import
    when multiple processes start simultaneously.
    """
    import matplotlib.pyplot as plt  # local import after backend selection

    for var, data in plot_data.items():
        df = pd.DataFrame(data)
        if df.empty:
            continue
        plt.figure(figsize=(10, 6))

        # Get the first column name dynamically
        time_col = df.columns[0]
        xlabel = time_col.split(' (')[0]  # Remove the parenthetical part if present

        if 'Observed' in df.columns:
            plt.plot(df[time_col], df['Observed'], label='Historical', marker='o')
        if 'Modeled' in df.columns:
            plt.plot(df[time_col], df['Modeled'], label='Modeled', marker='x')

        plt.title(f'{var}: Modeled vs Historical')
        plt.xlabel(xlabel)
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        out_path = os.path.join(individual_dir, f'{var}_comparison.png')
        plt.savefig(out_path)
        plt.close()


def handle_error(individual_dir, message, stdout="", stderr=""):
    report_data = {
        "status": "ERROR",
        "message": message,
        "stdout": stdout,
        "stderr": stderr
    }
    try:
        update_model_report(individual_dir, report_data)
    except Exception as e:
        log_message(f"Failed to write error report: {e}", TextColors.RED)

    log_message(message, TextColors.RED)
    return "FAILED", None


def run_model(individual_dir, test_args=None):
    pid = os.getpid()
    log_message(f"[PID {pid}] Running model for individual...", TextColors.YELLOW)
    log_message(f"[PID {pid}] Individual directory: {individual_dir}")

    # Remove old model compilations
    for file in ['model.dll', 'model.o']:
        file_path = os.path.join(individual_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            log_message(f"[PID {pid}] Removed old compilation: {file}", TextColors.YELLOW)

    # Build and run R script command
    individual_dir_fslash = individual_dir.replace('\\', '/')
    timestamp = int(time.time())

    cmd = [
        "Rscript", "--vanilla", "./Code/ControlFile.R",
        "--model_location", f"{individual_dir_fslash}",
        "--timestamp", str(timestamp),
    ]

    timeout_env = os.environ.get("RUN_MODEL_TIMEOUT_SEC", "").strip()
    timeout_sec = int(timeout_env) if timeout_env.isdigit() and int(timeout_env) > 0 else None

    log_message(f"[PID {pid}] Launching Rscript... (timeout={timeout_sec or 'none'})", TextColors.CYAN)
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_sec
        )
    except subprocess.TimeoutExpired as e:
        return handle_error(
            individual_dir,
            f"R script timed out after {timeout_sec}s",
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        )
    except Exception as e:
        return handle_error(
            individual_dir,
            f"Failed to launch R script: {e}"
        )

    stdout = (process.stdout or "").strip()
    stderr = (process.stderr or "").strip()

    # Print stdout/stderr for debugging (explicitly marked)
    if stdout:
        log_message(f"[PID {pid}] R script stdout BEGIN >>>", TextColors.YELLOW)
        log_message(stdout)
        log_message(f"[PID {pid}] R script stdout END <<<", TextColors.YELLOW)
    if stderr:
        log_message(f"[PID {pid}] R script stderr BEGIN >>>", TextColors.YELLOW)
        log_message(stderr)
        log_message(f"[PID {pid}] R script stderr END <<<", TextColors.YELLOW)

    # Check if R script failed or compilation failed
    if process.returncode != 0 or 'Compilation failed' in stdout or 'Compilation failed' in stderr:
        error_msg = (
            "Model failed to compile."
            if ('Compilation failed' in stdout or 'Compilation failed' in stderr)
            else f"R script failed with return code {process.returncode}"
        )
        return handle_error(individual_dir, error_msg, stdout, stderr)

    try:
        # --- Robust JSON extraction ---
        def extract_json(text: str):
            start = text.find('JSON_REPORT_START')
            if start == -1:
                return None

            # JSON could start on the SAME line or the NEXT line.
            # Prefer the first "{" after the marker.
            brace_start = text.find('{', start)
            if brace_start == -1:
                # Fallback to the next newline after marker
                newline_after = text.find('\n', start)
                if newline_after == -1:
                    return None
                brace_start = newline_after + 1

            end = text.find('JSON_REPORT_END', brace_start)
            if end == -1:
                return None

            return text[brace_start:end].strip()

        # Try stdout first, then stderr
        json_str = extract_json(stdout)
        if json_str is None:
            json_str = extract_json(stderr)
        if json_str is None or not json_str:
            return handle_error(individual_dir, "No JSON report found in output", stdout, stderr)

        log_message(f"[PID {pid}] JSON detected, parsing...", TextColors.CYAN)
        try:
            report_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            log_message(f"[PID {pid}] JSON parse error: {str(e)}", TextColors.RED)
            log_message("[PID {pid}] JSON string follows:", TextColors.YELLOW)
            log_message(json_str)
            return handle_error(individual_dir, f"Failed to parse JSON report: {str(e)}", stdout, stderr)

        # Add stdout/stderr to report for traceability
        report_data["stdout"] = stdout
        report_data["stderr"] = stderr

        # Persist the report
        log_message(f"[PID {pid}] Writing model report...", TextColors.CYAN)
        update_model_report(individual_dir, report_data)

        if report_data.get('status') == 'ERROR':
            return handle_error(individual_dir, f"Model execution failed: {report_data.get('message')}")

        # Extract objective value
        objective_value = report_data.get('objective_value')
        if objective_value is None:
            # Optional backward compatibility:
            # objective_value = report_data.get('model_report', {}).get('nll')
            raise ValueError("Objective function value is missing from the report")

        if isinstance(objective_value, list):
            if len(objective_value) == 0:
                raise ValueError("Objective function value list is empty")
            objective_value = objective_value[0]

        objective_value = float(objective_value)
        if objective_value == float('inf'):
            raise ValueError("Objective function value is infinity")

        log_message(f"[PID {pid}] Objective value: {objective_value}", TextColors.BLUE)

        # Remove any existing error files
        remove_error_files(individual_dir)

        # Plotting (optional; can be disabled)
        skip_plots = os.environ.get("SKIP_PLOTS", "").strip().lower() in {"1", "true", "yes", "y"}
        plot_data = report_data.get('plot_data', {})

        # Normalize list->dict if necessary
        if isinstance(plot_data, list):
            plot_data = {str(i): data for i, data in enumerate(plot_data)}

        if not skip_plots and plot_data:
            log_message(f"[PID {pid}] Generating plots...", TextColors.CYAN)
            try:
                plot_model_vs_historical(individual_dir, plot_data)
            except Exception as pe:
                # Don't fail the whole run on plotting issues in parallel mode
                log_message(f"[PID {pid}] Plotting failed (non-fatal): {pe}", TextColors.YELLOW)
        else:
            if skip_plots:
                log_message(f"[PID {pid}] SKIP_PLOTS=1 -> skipping plotting.", TextColors.YELLOW)
            else:
                log_message(f"[PID {pid}] No plot_data -> skipping plotting.", TextColors.YELLOW)

        log_message(f"[PID {pid}] Objective Value: {objective_value}", TextColors.BLUE)
        log_message('[PID {}] Successful model run'.format(pid), TextColors.GREEN)
        log_message('----------------------------------------------')

        return "SUCCESS", objective_value

    except Exception as e:
        return handle_error(individual_dir, f"Error processing model report: {str(e)}", stdout, stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log_message("Please provide the individual directory as an argument.", TextColors.RED)
        sys.exit(1)

    individual_dir = sys.argv[1]
    status, result = run_model(individual_dir)
    log_message(f"Final status: {status}")
    log_message(f"Final result: {result}")
    if status == "FAILED":
        log_message(f"Error information: {result}", TextColors.RED)
