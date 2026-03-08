import os

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class DockerMount:
    source: Path
    target: Path
    mode: str = "rw"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_proxy_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def build_docker_run_cmd(
    image_name: str,
    container_cmd: Sequence[str],
    workdir: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
    mounts: Optional[Sequence[DockerMount]] = None,
    use_gpu: bool = True,
    ipc_host: bool = False,
    network_host: bool = True,
    run_as_host_user: bool = True,
) -> List[str]:
    if not image_name:
        raise ValueError("image_name is required")
    if not container_cmd:
        raise ValueError("container_cmd is required")

    cmd = ["docker", "run", "--rm"]

    if use_gpu:
        cmd += ["--gpus", "all"]
    if ipc_host:
        cmd += ["--ipc", "host"]
    if network_host:
        cmd += ["--network", "host"]
    if run_as_host_user:
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]

    merged_env = {
        "HOME": "/tmp/aiws-docker-home",
        "XDG_CACHE_HOME": "/tmp/aiws-xdg-cache",
        "PYTHONUNBUFFERED": "1",
    }
    merged_env.update(default_proxy_env())
    if env:
        merged_env.update({k: v for k, v in env.items() if v is not None})

    merged_mounts = [
        DockerMount(source=repo_root(), target=repo_root(), mode="rw"),
    ]
    if mounts:
        merged_mounts.extend(mounts)

    mount_map: Dict[str, DockerMount] = {}
    for mount in merged_mounts:
        mount_map[str(Path(mount.target))] = mount

    for mount in mount_map.values():
        source = Path(mount.source).resolve()
        target = Path(mount.target)
        cmd += ["-v", f"{source}:{target}:{mount.mode}"]

    for key, value in merged_env.items():
        cmd += ["-e", f"{key}={value}"]

    if workdir is not None:
        cmd += ["-w", str(workdir)]

    cmd.append(image_name)
    cmd.extend(container_cmd)
    return cmd
