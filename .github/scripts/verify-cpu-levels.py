#!/usr/bin/env python3
"""Check whisper-server's builds for processors without AVX2 (OpenWhispr/openwhispr#2356).

Each build runs under an emulator that implements only its CPU level: Intel SDE
(-ivb, -snb) on Windows, QEMU user mode (-cpu IvyBridge, -cpu SandyBridge) on
Linux. The levels are upstream ggml's CPU variants:

  ivybridge    SSE4.2 + AVX + F16C (Intel Ivy Bridge, AMD Piledriver, Jaguar)
  sandybridge  SSE4.2 + AVX        (Intel Sandy Bridge, AMD Bulldozer)

1. Each build starts the way OpenWhispr starts it (a real model, Silero VAD on),
   transcribes samples/jfk.wav, and reports exactly its level's instruction sets.
2. The next build up must be stopped before it starts serving: the primary
   (AVX2) build on Ivy Bridge, where ggml's startup code faults, and the
   ivybridge build on Sandy Bridge, where its level check (OPENWHISPR_CPU_LEVEL_CHECK,
   examples/server/cpu-level-check.cpp) refuses: ggml alone only reaches F16C at
   the first transcription. That proves the emulator enforces the level, so check
   1 means something, and it proves what OpenWhispr's fallback relies on: a build
   dies at startup on a processor it does not support, not mid-transcription.

--with-llama-libraries also starts each build on its own level with llama.cpp's
libraries beside it, as in OpenWhispr's resources/bin: whisper-server loads
every ggml library in its folder at startup (ggml_backend_load_all).

--emulator none runs everything natively. It exists to smoke-test this script
locally and can never pass: nothing stops the rejection checks.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIO = REPO / "samples" / "jfk.wav"
EXPECTED_WORDS = "ask not what your country can do for you"

DOWNLOADS = {
    # models/README.md lists sha1 bd577a113a864445d4c299885e0cb97d4ba92b5f for this file
    "model": (
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    ),
    # The Silero VAD model OpenWhispr ships for its whisper-server
    "vad": (
        "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
        "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    ),
    # Intel SDE 9.58. Intel's download mirror refuses non-browser clients; this
    # mirror serves the identical file (oven-sh/bun pins the same sha256).
    "sde": (
        "https://github.com/petarpetrovt/setup-sde/releases/download/binaries/"
        "sde-external-9.58.0-2025-06-16-win.tar.xz",
        "ebb8b3b63fcb0b6c1f9721118ba4883703d2aed9e0db2defed4e44fba78d9ca9",
    ),
    # The llama.cpp CPU builds OpenWhispr bundles (LLAMA_CPP_TAG in its
    # scripts/download-llama-server.js); used by --with-llama-libraries
    "llama-win32": (
        "https://github.com/ggml-org/llama.cpp/releases/download/b9763/"
        "llama-b9763-bin-win-cpu-x64.zip",
        "05144ee4d885a778ebaef619f79ca0b8a4edb7f017eaf70086a8781ff003815f",
    ),
    "llama-linux": (
        "https://github.com/ggml-org/llama.cpp/releases/download/b9763/"
        "llama-b9763-bin-ubuntu-x64.tar.gz",
        "4bd11fe0cea35223b240496062900ed9493b46f20a08747d431bfdc2252af2d8",
    ),
}

EMULATED_CPUS = {
    "ivybridge": {"sde": "-ivb", "qemu": "IvyBridge"},
    "sandybridge": {"sde": "-snb", "qemu": "SandyBridge"},
}
# What each build's system_info line must report, among the instruction sets that matter here
EXPECTED_FEATURES = {"ivybridge": {"AVX", "F16C"}, "sandybridge": {"AVX"}}
CHECKED_FEATURES = {"AVX", "AVX2", "F16C", "FMA", "BMI2", "AVX512"}
# The libraries OpenWhispr copies out of the llama.cpp archive (copyLibraries there)
LIBRARY = re.compile(r"\.(dll|so(\.\d+)*)$")

SDE_VIOLATION = re.compile(r"SDE-ERROR:.*not valid for specified chip.*", re.IGNORECASE)
# What cpu-level-check.cpp prints before it raises the fault
LEVEL_CHECK = re.compile(r"whisper-server: built for \w+, which this processor does not support")
STATUS_ILLEGAL_INSTRUCTION = 0xC000001D
READY_TIMEOUT_S = 900
INFERENCE_TIMEOUT_S = 2700


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name, work):
    url, expected = DOWNLOADS[name]
    dest = work / url.rsplit("/", 1)[-1]
    if not dest.exists() or sha256_of(dest) != expected:
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
    actual = sha256_of(dest)
    if actual != expected:
        raise SystemExit(f"{dest.name}: sha256 {actual}, expected {expected}")
    return dest


def extract(archive, dest):
    shutil.rmtree(dest, ignore_errors=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        return
    with tarfile.open(archive) as t:
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")
        else:
            t.extractall(dest)


def unpack_server(zip_path, dest):
    """Extract a release zip and return the one whisper-server executable in it."""
    extract(zip_path, dest)
    servers = [p for p in dest.iterdir() if p.name.startswith("whisper-server")]
    if len(servers) != 1:
        raise SystemExit(f"{zip_path.name}: expected one whisper-server, found {servers}")
    servers[0].chmod(0o755)  # zipfile drops the executable bit
    return servers[0]


def with_llama_libraries(server, level, work):
    """Copy a build's folder and llama.cpp's libraries into one folder, flat, the
    way OpenWhispr lays out resources/bin: whisper.cpp first, then llama.cpp,
    whose files win a name clash. Returns the copied server."""
    llama = work / "llama"
    if not llama.exists():
        extract(fetch("llama-win32" if os.name == "nt" else "llama-linux", work), llama)
    dest = work / f"with-llama-{level}"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    for path in server.parent.iterdir():
        shutil.copy2(path, dest / path.name)
    for path in llama.rglob("*"):
        if path.is_file() and LIBRARY.search(path.name):
            shutil.copy2(path, dest / path.name)
    return dest / server.name


def emulator(kind, cpu, work):
    """Return (command prefix, working directory) that runs a program on `cpu`."""
    if kind == "sde":
        root = work / "sde"
        if not root.exists():
            extract(fetch("sde", work), root)
        sde = next(root.glob("sde-external-*/sde.exe"))
        # SDE must run from its own directory to find Pin's DLLs
        return [str(sde), EMULATED_CPUS[cpu]["sde"], "--"], sde.parent
    if kind == "qemu":
        return ["qemu-x86_64", "-cpu", EMULATED_CPUS[cpu]["qemu"]], None
    return [], None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(prefix, cwd, server, model, vad_model, log_path):
    """Start whisper-server with the arguments OpenWhispr passes (buildWhisperServerArgs)."""
    port = free_port()
    args = [str(server), "--model", str(model), "--host", "127.0.0.1", "--port", str(port)]
    args += ["--language", "auto", "--max-len", "4096"]
    if vad_model:
        args += ["--vad", "--vad-model", str(vad_model)]
    log = open(log_path, "wb")
    proc = subprocess.Popen(prefix + args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, port


def wait_ready(proc, port):
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5):
                return True
        except urllib.error.HTTPError:
            return True  # any HTTP answer means it is up, as OpenWhispr's health check treats it
        except OSError:
            time.sleep(1)
    return False


def transcribe(port):
    """POST samples/jfk.wav to /inference with the fields OpenWhispr sends."""
    boundary = uuid.uuid4().hex
    fields = {
        "language": "auto",
        "entropy_thold": "2.8",
        "logprob_thold": "-1.25",
        "response_format": "json",
    }
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    ]
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"jfk.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n".encode()
        + AUDIO.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/inference",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=INFERENCE_TIMEOUT_S) as response:
        return json.loads(response.read())["text"]


def stop(proc):
    if proc.poll() is None:
        if os.name == "nt":
            # SDE runs the server as a child process; /T ends the whole tree
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def violation(kind, returncode, output):
    """How the program was stopped for an instruction the emulated CPU lacks: SDE's
    chip check, or an illegal-instruction death (a real fault, or the one a level
    check raises after naming what is missing), or None."""
    if kind == "sde":
        match = SDE_VIOLATION.search(output)
        if match:
            return match.group(0).strip()
    illegal = (kind == "sde" and returncode == STATUS_ILLEGAL_INSTRUCTION) or (
        kind == "qemu" and returncode == -signal.SIGILL
    )
    if not illegal:
        return None
    refusals = LEVEL_CHECK.findall(output)
    return "; ".join(refusals) if refusals else "killed by an illegal instruction"


def reported_features(output):
    """Instruction sets the build says it was compiled for (its system_info line)."""
    for line in output.splitlines():
        if line.startswith("system_info:") and "CPU :" in line:
            enabled = re.findall(r"(\w+) = 1", line.split("CPU :", 1)[1])
            return set(enabled) & CHECKED_FEATURES
    return None


def words(text):
    return " ".join(re.sub(r"[^a-z]+", " ", text.lower()).split())


def run(kind, level, server, model, vad_model, work, label, transcribe_audio):
    """Start `server` on an emulated `level` processor. Returns (ready, text, output, returncode)."""
    prefix, cwd = emulator(kind, level, work)
    log_path = work / f"{label.replace(' ', '-')}.log"
    proc, log, port = start_server(prefix, cwd, server, model, vad_model, log_path)
    ready, text = False, None
    try:
        ready = wait_ready(proc, port)
        if ready and transcribe_audio:
            text = transcribe(port)
    finally:
        stop(proc)
        log.close()
    return ready, text, log_path.read_text(errors="replace"), proc.returncode


def check_transcribes(kind, level, server, model, vad_model, work):
    label = f"{server.name} on {level}"
    ready, text, output, returncode = run(kind, level, server, model, vad_model, work, label, True)
    hit = violation(kind, returncode, output)
    if hit:
        return False, f"{label} ran an instruction {level} lacks: {hit}"
    if not ready or text is None:
        return False, f"{label} did not start (exit {returncode}):\n{output[-2000:]}"
    if EXPECTED_WORDS not in words(text):
        return False, f"{label} transcribed jfk.wav as {text!r}"
    features = reported_features(output)
    if features != EXPECTED_FEATURES[level]:
        expected = sorted(EXPECTED_FEATURES[level])
        return False, f"{label} reports {sorted(features or [])}, expected {expected}"
    return True, f"{label}: transcribed {text.strip()!r}, reports {sorted(features)}"


def check_rejected(kind, level, server, model, work, by_level_check):
    label = f"{server.name} on {level}"
    ready, _, output, returncode = run(kind, level, server, model, None, work, label, False)
    if ready:
        return False, (
            f"{label} started serving. Either the emulator does not enforce {level} (then the "
            "transcription checks prove nothing) or the build no longer dies at startup on a "
            "processor it does not support (then OpenWhispr's fallback cannot see the crash)"
        )
    hit = violation(kind, returncode, output)
    if not hit:
        return False, f"{label} exited ({returncode}) without an instruction violation"
    if by_level_check and not LEVEL_CHECK.search(output):
        return False, f"{label} was stopped ({hit}), but not by its level check"
    return True, f"{label}: stopped at startup as expected ({hit})"


def check_starts_with_llama(kind, level, server, model, work):
    label = f"{server.name} with llama.cpp libraries on {level}"
    ready, _, output, returncode = run(kind, level, server, model, None, work, label, False)
    hit = violation(kind, returncode, output)
    if hit:
        return False, f"{label} ran an instruction {level} lacks: {hit}"
    if not ready:
        return False, f"{label} did not start (exit {returncode}):\n{output[-2000:]}"
    return True, f"{label}: started"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emulator", choices=["sde", "qemu", "none"], required=True)
    parser.add_argument("--primary-zip", type=Path, required=True)
    parser.add_argument("--ivybridge-zip", type=Path, required=True)
    parser.add_argument("--sandybridge-zip", type=Path, required=True)
    parser.add_argument("--with-llama-libraries", action="store_true")
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    model, vad_model = fetch("model", work), fetch("vad", work)
    primary = unpack_server(args.primary_zip.resolve(), work / "primary")
    levels = {
        "ivybridge": unpack_server(args.ivybridge_zip.resolve(), work / "ivybridge"),
        "sandybridge": unpack_server(args.sandybridge_zip.resolve(), work / "sandybridge"),
    }

    kind = args.emulator
    results = [
        check_transcribes(kind, "ivybridge", levels["ivybridge"], model, vad_model, work),
        check_transcribes(kind, "sandybridge", levels["sandybridge"], model, vad_model, work),
        check_rejected(kind, "ivybridge", primary, model, work, by_level_check=False),
        check_rejected(kind, "sandybridge", levels["ivybridge"], model, work, by_level_check=True),
    ]
    if args.with_llama_libraries:
        for level, server in levels.items():
            server = with_llama_libraries(server, level, work)
            results.append(check_starts_with_llama(kind, level, server, model, work))
    for ok, message in results:
        print(f"{'PASS' if ok else 'FAIL'}: {message}", flush=True)
    sys.exit(0 if all(ok for ok, _ in results) else 1)


if __name__ == "__main__":
    main()
