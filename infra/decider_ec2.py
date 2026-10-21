"""Run the Strands Decider 2B server on an EC2 GPU instance, reachable only from your IP.

The same instance also runs Ollama with a small open chat model and an embedding model, so
Tact works on accounts where Amazon Bedrock is not available (TACT_PROVIDER=ollama).

    uv run python infra/decider_ec2.py up      # launch (or start) and wait until it answers
    uv run python infra/decider_ec2.py app     # also run the Tact demo app on the instance
    uv run python infra/decider_ec2.py status  # state, URL, health
    uv run python infra/decider_ec2.py stop    # stop billing for compute, keep the disk
    uv run python infra/decider_ec2.py down    # terminate everything it created

Writes the server URLs to .cache/decider_url and .cache/ollama_url so the app and the eval
pick them up. The servers have no authentication, so the security group admits only your
public IP.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
URL_FILE = CACHE / "decider_url"
OLLAMA_FILE = CACHE / "ollama_url"
APP_FILE = CACHE / "app_url"
KEY_FILE = CACHE / "tact-decider.pem"

REGION = os.environ.get("AWS_REGION", "us-east-1")
NAME = "tact-decider"
PORT = 8099
OLLAMA_PORT = 11434
APP_PORT = 8501
MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
GPU_TYPE = os.environ.get("TACT_GPU_TYPE", "g6.xlarge")
CPU_TYPE = os.environ.get("TACT_CPU_TYPE", "c7i.2xlarge")
SMALL_CPU_TYPE = "m7i.xlarge"  # 4 vCPU / 16 GB: fits the default vCPU quota of new accounts
AMI_PARAM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"

USER_DATA = """#!/bin/bash
set -eux
exec > /var/log/decider-setup.log 2>&1
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
/root/.local/bin/uv venv --python 3.12 /opt/decider
/root/.local/bin/uv pip install --python /opt/decider/bin/python strands-decider
DEVICE=cpu
if nvidia-smi; then DEVICE=cuda; fi
cat > /etc/systemd/system/decider.service <<EOF
[Unit]
After=network-online.target
[Service]
Environment=HF_HOME=/opt/hf
ExecStart=/opt/decider/bin/strands-decider serve {model} --host 0.0.0.0 --port {port} --device $DEVICE
Restart=always
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now decider

# Fallback LLM for accounts without Bedrock: Ollama, listening on all interfaces.
curl -fsSL https://ollama.com/install.sh | sh
mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]
Environment=OLLAMA_HOST=0.0.0.0:{ollama_port}
Environment=OLLAMA_KEEP_ALIVE=24h
' >/etc/systemd/system/ollama.service.d/host.conf
systemctl daemon-reload
systemctl restart ollama
sleep 5
CHAT=qwen2.5:3b-instruct
if [ "$DEVICE" = cuda ]; then CHAT=qwen2.5:7b-instruct; fi
export OLLAMA_HOST=127.0.0.1:{ollama_port}
ollama pull nomic-embed-text
ollama pull $CHAT
ollama cp $CHAT tact-chat
""".format(model=MODEL, port=PORT, ollama_port=OLLAMA_PORT)

ec2 = boto3.client("ec2", region_name=REGION)


def my_ip() -> str:
    return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()


def find_instance() -> dict | None:
    res = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [NAME]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    for r in res["Reservations"]:
        for i in r["Instances"]:
            return i
    return None


def security_group(ip: str) -> str:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    groups = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [NAME]},
                                                   {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if groups:
        sg = groups[0]["GroupId"]
    else:
        sg = ec2.create_security_group(GroupName=NAME, Description="Tact decider, owner IP only",
                                       VpcId=vpc)["GroupId"]
    for port in (PORT, OLLAMA_PORT, APP_PORT, 22):
        try:
            ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": port, "ToPort": port,
                "IpRanges": [{"CidrIp": f"{ip}/32", "Description": "owner"}]}])
        except ClientError as e:
            if "Duplicate" not in str(e):
                raise
    return sg


def key_pair() -> str:
    CACHE.mkdir(exist_ok=True)
    try:
        ec2.describe_key_pairs(KeyNames=[NAME])
    except ClientError:
        key = ec2.create_key_pair(KeyName=NAME, KeyType="ed25519")
        KEY_FILE.write_text(key["KeyMaterial"], newline="\n")  # ssh rejects CRLF keys
    return NAME


def launch() -> dict:
    ip = my_ip()
    sg = security_group(ip)
    ami = boto3.client("ssm", region_name=REGION).get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    for itype in (GPU_TYPE, CPU_TYPE, SMALL_CPU_TYPE):
        try:
            print(f"launching {itype} from {ami} (access limited to {ip})")
            res = ec2.run_instances(
                ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
                KeyName=key_pair(), SecurityGroupIds=[sg], UserData=USER_DATA,
                BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                      "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}}],
                TagSpecifications=[{"ResourceType": "instance",
                                    "Tags": [{"Key": "Name", "Value": NAME}]}],
            )
            return res["Instances"][0]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            print(f"  {itype} failed: {code}: {e.response['Error']['Message']}")
            if code not in ("VcpuLimitExceeded", "InsufficientInstanceCapacity", "Unsupported",
                            "InvalidParameterCombination"):
                raise
    raise SystemExit("could not launch a GPU or CPU instance")


def health(url: str) -> dict | None:
    try:
        return json.loads(urllib.request.urlopen(f"{url}/health", timeout=5).read())
    except Exception:
        return None


def ollama_models(url: str) -> list[str]:
    try:
        tags = json.loads(urllib.request.urlopen(f"{url}/api/tags", timeout=5).read())
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return []


def wait_ollama(url: str, deadline: float) -> None:
    """The fallback LLM installs after the decider; only needed when Bedrock is unavailable."""
    while time.time() < deadline:
        if any(n.startswith("tact-chat") for n in ollama_models(url)):
            print("ollama ready:", url)
            return
        time.sleep(15)
        print("  ...ollama still pulling models (only needed without Bedrock)", flush=True)
    print("ollama not ready yet; check later with: infra/decider_ec2.py status")


def up() -> None:
    inst = find_instance()
    if inst is None:
        inst = launch()
    elif inst["State"]["Name"] in ("stopped", "stopping"):
        ec2.get_waiter("instance_stopped").wait(InstanceIds=[inst["InstanceId"]])
        security_group(my_ip())  # your IP may have changed since the last run
        ec2.start_instances(InstanceIds=[inst["InstanceId"]])
        print("starting stopped instance")
    iid = inst["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    inst = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    url = f"http://{inst['PublicIpAddress']}:{PORT}"
    CACHE.mkdir(exist_ok=True)
    URL_FILE.write_text(url)
    OLLAMA_FILE.write_text(f"http://{inst['PublicIpAddress']}:{OLLAMA_PORT}")
    print(f"{iid} ({inst['InstanceType']}) running at {url}; waiting for the model to load")
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        h = health(url)
        if h:
            print("decider ready:", json.dumps(h))
            wait_ollama(OLLAMA_FILE.read_text(), deadline)
            return
        time.sleep(15)
        print("  ...still installing / downloading the model", flush=True)
    raise SystemExit(f"server did not answer in 30 min; ssh -i {KEY_FILE} ubuntu@{inst['PublicIpAddress']} "
                     "and read /var/log/decider-setup.log")


# The demo app runs next to the servers it calls, as the ubuntu user, from a copy of the
# current git HEAD (no push needed). Same owner-IP-only security group.
APP_UNIT = """[Unit]
After=network-online.target decider.service ollama.service
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/tact
Environment=TACT_DECIDER_URL=http://127.0.0.1:{port}
Environment=TACT_OLLAMA_URL=http://127.0.0.1:{ollama_port}
Environment=TACT_PROVIDER={provider}
Environment=TACT_CACHE=/home/ubuntu/tact-cache
ExecStart=/home/ubuntu/.local/bin/uv run --frozen --no-dev streamlit run app.py --server.port {app_port} \
  --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=always
[Install]
WantedBy=multi-user.target
"""


def ssh(ip: str, command: str, stdin: bytes | None = None) -> None:
    subprocess.run(["ssh", "-i", str(KEY_FILE), "-o", "StrictHostKeyChecking=accept-new",
                    f"ubuntu@{ip}", command], input=stdin, check=True)


def app() -> None:
    inst = find_instance()
    if not inst or inst["State"]["Name"] != "running":
        raise SystemExit("start the instance first: infra/decider_ec2.py up")
    ip = inst["PublicIpAddress"]
    security_group(my_ip())
    provider = os.environ.get("TACT_PROVIDER") or (
        (CACHE / "provider").read_text().strip() if (CACHE / "provider").exists() else "bedrock")
    print(f"copying git HEAD to {ip}")
    code = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                          capture_output=True, check=True).stdout
    ssh(ip, "rm -rf ~/tact && mkdir -p ~/tact ~/tact-cache && tar -x -C ~/tact", code)
    print("installing (first time: a few minutes)")
    ssh(ip, "test -x ~/.local/bin/uv || curl -LsSf https://astral.sh/uv/install.sh | sh; "
            "cd ~/tact && ~/.local/bin/uv sync --frozen --no-dev -q")
    unit = APP_UNIT.format(port=PORT, ollama_port=OLLAMA_PORT, app_port=APP_PORT, provider=provider)
    ssh(ip, "sudo tee /etc/systemd/system/tact-app.service >/dev/null && sudo systemctl daemon-reload "
            "&& sudo systemctl enable -q tact-app && sudo systemctl restart tact-app", unit.encode())
    url = f"http://{ip}:{APP_PORT}"
    for _ in range(40):
        try:
            if urllib.request.urlopen(f"{url}/_stcore/health", timeout=5).read().strip() == b"ok":
                APP_FILE.write_text(url)
                print(f"Tact is running on AWS: {url}  (provider: {provider}; open from your IP only)")
                return
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"app did not start; ssh -i {KEY_FILE} ubuntu@{ip} journalctl -u tact-app")


def status() -> None:
    inst = find_instance()
    if not inst:
        print("no instance")
        return
    url = URL_FILE.read_text() if URL_FILE.exists() else "?"
    running = inst["State"]["Name"] == "running"
    print(inst["InstanceId"], inst["InstanceType"], inst["State"]["Name"], url,
          health(url) if running else "")
    if running and OLLAMA_FILE.exists():
        print("ollama:", OLLAMA_FILE.read_text(), ollama_models(OLLAMA_FILE.read_text()))
    if running and APP_FILE.exists():
        print("app:", APP_FILE.read_text())


def stop() -> None:
    inst = find_instance()
    if inst:
        ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
        print("stopping", inst["InstanceId"])


def down() -> None:
    inst = find_instance()
    if inst:
        ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[inst["InstanceId"]])
        print("terminated", inst["InstanceId"])
    for fn in (lambda: ec2.delete_key_pair(KeyName=NAME),
               lambda: ec2.delete_security_group(GroupName=NAME)):
        try:
            fn()
        except ClientError as e:
            print("cleanup:", e)
    for f in (URL_FILE, OLLAMA_FILE, APP_FILE):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    {"up": up, "app": app, "status": status, "stop": stop, "down": down}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
