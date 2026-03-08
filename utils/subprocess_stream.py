import subprocess
import sys
from typing import Sequence


def run_subprocess_stream(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    """Run a subprocess while streaming combined stdout/stderr to the terminal."""
    process = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    returncode = process.wait()
    stdout = "".join(output_lines)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(cmd), output=stdout, stderr="")

    return subprocess.CompletedProcess(list(cmd), returncode, stdout, "")
