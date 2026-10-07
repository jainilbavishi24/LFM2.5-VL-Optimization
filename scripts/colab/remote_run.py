"""Executed inside the Colab kernel by scripts/colab/colab.sh: runs $CMD in bash from the project dir,
streaming output live. Raises on non-zero exit so the local side sees the failure."""
import os
import subprocess

cmd = os.environ["CMD"]
print(f"[remote] $ {cmd}", flush=True)
p = subprocess.Popen(["bash", "-lc", cmd], cwd=os.environ.get("REMOTE_DIR", "/content/PAA_Project"),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"[remote] exit code {rc}", flush=True)
if rc != 0:
    raise RuntimeError(f"remote command failed with exit code {rc}")
