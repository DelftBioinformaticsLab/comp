"""Execute one figure notebook in place, one execution at a time (concurrent writes to the network drive fail).

    python run_nb.py fig3.ipynb      (from src/figures)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, ".nb_running")


def main():
    nb = sys.argv[1]
    os.environ["PYTHONUTF8"] = "1"
    os.environ.pop("MPLBACKEND", None)
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > 1800:  # a stale lock from a crashed run
                    os.rmdir(LOCK)
                    continue
            except FileNotFoundError:                            # released between mkdir and getmtime: retry
                continue
            time.sleep(5)
    try:
        print(f"lock after {time.time() - t0:.0f} s; executing {nb}", flush=True)
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=1200", nb], cwd=HERE)
        sys.exit(r.returncode)
    finally:
        os.rmdir(LOCK)


if __name__ == "__main__":
    main()
