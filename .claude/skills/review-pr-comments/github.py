"""Read a PR's review feedback and post replies, without the gh CLI.

    python .claude/skills/review-pr-comments/github.py fetch 31
    python .claude/skills/review-pr-comments/github.py reply 31 replies.json

``replies.json`` maps an inline review comment id to the reply text; the key ``"pr"`` posts a
top-level PR comment instead. Writes use the token Git Credential Manager stores for github.com,
which is never printed.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import requests

API = "https://api.github.com/repos/CarlosHeinze/MLOps_Team2"


def token() -> str:
    credential = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GCM_INTERACTIVE": "never"},
    ).stdout
    return dict(line.split("=", 1) for line in credential.splitlines() if "=" in line)["password"]


def get(path: str, headers: dict | None = None) -> list | dict:
    response = requests.get(f"{API}/{path}", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch(pr: str) -> None:
    pull = get(f"pulls/{pr}")
    print(f"# PR #{pr}: {pull['title']} ({pull['state']}, head {pull['head']['ref']})\n")
    for review in get(f"pulls/{pr}/reviews"):
        if review["body"]:
            print(f"## Review {review['id']} by {review['user']['login']} ({review['state']})")
            print(review["body"], "\n")
    for comment in get(f"pulls/{pr}/comments?per_page=100"):
        where = f"{comment['path']}:{comment.get('line') or comment.get('original_line')}"
        reply = f", reply to {comment['in_reply_to_id']}" if comment.get("in_reply_to_id") else ""
        print(f"## Inline {comment['id']} by {comment['user']['login']} at {where}{reply}")
        print(comment["body"], "\n")
    for comment in get(f"issues/{pr}/comments?per_page=100"):
        print(f"## Conversation {comment['id']} by {comment['user']['login']}")
        print(comment["body"], "\n")


def reply(pr: str, replies_file: str) -> None:
    headers = {"Authorization": f"Bearer {token()}", "Accept": "application/vnd.github+json"}
    for key, body in json.loads(Path(replies_file).read_text(encoding="utf-8")).items():
        url = (
            f"{API}/issues/{pr}/comments"
            if key == "pr"
            else f"{API}/pulls/{pr}/comments/{key}/replies"
        )
        response = requests.post(url, headers=headers, json={"body": body}, timeout=30)
        response.raise_for_status()
        print(f"{key}: {response.json()['html_url']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console code page cannot print emoji
    command, *args = sys.argv[1:]
    {"fetch": fetch, "reply": reply}[command](*args)
