---
name: ntfy-notify
description: Two-way phone messaging with the user (msinclair) via ntfy when they are away from the terminal. SEND a push notification (long job finished, a result worth knowing now, a blocking decision) and, when you need an answer, READ their reply back. Works from any session/project. Do NOT use for routine progress chatter or when the user is clearly at the terminal (use the in-terminal PushNotification tool for that).
---

# ntfy-notify

Reach the user (msinclair) on their phone even when they're away from this session — and read their replies back.

## How to send

Run the helper script:

```
~/.claude/ntfy-notify.sh "your message here"
```

With options:

```
~/.claude/ntfy-notify.sh -t "Title" -p high -g white_check_mark,tada "message"
```

- `-t TITLE` — notification title
- `-p PRIORITY` — `min` | `low` | `default` | `high` | `max` (or `1`..`5`)
- `-g TAGS` — comma-separated emoji short-codes (e.g. `warning`, `white_check_mark`, `rotating_light`)
- `-c URL` — a tap-to-open URL

The script auto-reads the user's ntfy access token from `~/.claude/.ntfy-token` (owner-only) and
sends it as `Authorization: Bearer`. A 200 response with message JSON means it was accepted.

## Reading the user's replies (two-way)

The user can reply from their phone, and you can read it by polling the same topic.

```
~/.claude/ntfy-read.sh                              # print cached messages (last 12h) as JSON lines
~/.claude/ntfy-read.sh --since 30m                  # only messages in the last 30 min
~/.claude/ntfy-read.sh --wait --since <ID|epoch> --exclude <yourMsgId> --timeout 10m
```

Bounded blocking wait (so an un-answered question never hangs you):
- `--timeout DUR` — how long to block before giving up. Accepts `90`, `90s`, `10m`, `1h` (default `300` = 5m).
  ALWAYS pass a bounded `--timeout` sized to the situation (e.g. `2m` for a quick yes/no, `15m` if the user
  said they'd step away briefly). Don't wait indefinitely.
- `--interval DUR` — poll frequency (default `4s`).
- On a reply: prints the message text, exit `0`.
- On no reply in time: prints `NO_REPLY: user did not reply within <N>s` to stderr, exit `3`. Detect this and
  move on gracefully (proceed with a safe default, or tell the user you'll wait for them next session).

Typical ask-and-wait flow:
1. Send the question with `ntfy-notify.sh`; capture the returned message `id` and `time` from its JSON.
2. Launch `ntfy-read.sh --wait --since <time> --exclude <id> --timeout <DUR>` as a BACKGROUND task. It polls
   until a new message (not your own) arrives OR the timeout elapses, then exits — which re-invokes you with
   the answer (exit 0) or the NO_REPLY signal (exit 3).
3. This only works while THIS session stays alive (you are not a persistent listener between turns). If
   the user might reply much later, tell them so, or use a scheduled/looping agent instead.

Messages the user sends have no `title`; messages you send have a `title` — a simple way to tell them apart.

### How the user sends a reply (important)
The ntfy iOS app does NOT offer a clean custom-message compose field — its "send" is basically a canned
test notification. So the user replies via one of:
- An iOS **Shortcut** that POSTs JSON `{"topic":"msinclair-c1793h6qyo3dyo664oul","message":"<text>"}` to
  `https://ntfy.sh/` (a "Reply to Claude" button — the reliable path).
- The **ntfy web app** at `https://ntfy.sh/app` (publish/send dialog).
Confirmed working end-to-end (asked a question, read a custom reply back).

## Raw fallback (if the script is missing)

```
curl -d "message" https://ntfy.sh/msinclair-c1793h6qyo3dyo664oul
```

## Config facts

- Server: `https://ntfy.sh` (public server)
- Topic: `msinclair-c1793h6qyo3dyo664oul`  ← EXACT string; note `664`, not `644`. The topic name is the only secret (free account, no reserved topic) — do not share it.
- Token file: `~/.claude/.ntfy-token` (chmod 600; token NOT stored in this skill or in memory)

## When to use vs not

USE: long-running job completed while the user may have walked away; a finding/result they'd want
immediately; a blocking decision you need before continuing; or when the user explicitly asks to be notified.

DON'T: routine progress updates; answering something they're clearly watching for in real time; or when
the user is active at the terminal (their output already reaches them — prefer the built-in PushNotification
tool, which is for pulling attention within the terminal/desktop).
