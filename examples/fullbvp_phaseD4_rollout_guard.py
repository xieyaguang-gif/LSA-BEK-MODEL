"""Event-driven D4 tool audit; inotify blocks until rollout content changes."""
import argparse
import collections
import ctypes
import datetime
import json
import os
from pathlib import Path
import re
import select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rollout', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    counts = collections.Counter()
    short = []
    status_reads = []
    windows = collections.deque()
    alerts = []
    complete = False
    if args.watch:
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_NONBLOCK)
        if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(args.rollout), 2 | 8) < 0:
            raise OSError(ctypes.get_errno(), 'inotify setup failed')
    with args.rollout.open() as source:
        while True:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.endswith('\n'):
                    source.seek(offset)
                    break
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                payload = entry.get('payload', {})
                if entry.get('type') == 'event_msg' and payload.get('type') == 'task_started':
                    complete = False
                if entry.get('type') == 'event_msg' and payload.get('type') == 'task_complete':
                    complete = True
                if entry.get('type') != 'response_item' or payload.get('type') not in ('function_call', 'custom_tool_call'):
                    continue
                name = payload.get('name', '')
                counts[name] += 1
                body = payload.get('arguments', payload.get('input', ''))
                if not isinstance(body, str):
                    body = json.dumps(body)
                stamp = entry.get('timestamp', '')
                found = [int(x) for x in re.findall(r'["\']?yield_time_ms["\']?\s*:\s*(\d+)', body)]
                is_wait = name.endswith(('.wait', '.write_stdin', '.wait_agent'))
                if any(x < 1800000 for x in found) or (is_wait and not found):
                    issue = dict(timestamp=stamp, tool=name, explicit_yields_ms=found)
                    short.append(issue)
                    try:
                        seconds = datetime.datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
                        windows.append(seconds)
                        while windows and seconds - windows[0] > 1800:
                            windows.popleft()
                        if len(windows) > 5:
                            alerts.append(dict(timestamp=stamp, short_calls_in_30min=len(windows)))
                    except ValueError:
                        pass
                if re.search(r'\b(?:tail|ps|nvidia-smi)\b', body):
                    status_reads.append(dict(timestamp=stamp, tool=name))
            data = dict(rollout=str(args.rollout), mode='inotify_events' if args.watch else 'single_snapshot',
                        task_complete_seen=complete, tool_calls=dict(counts),
                        total_tool_calls=sum(counts.values()), short_or_unspecified_wait_calls=short,
                        status_command_candidates=status_reads, alerts=alerts)
            args.output.write_text(json.dumps(data, indent=2) + '\n')
            if not args.watch or complete:
                break
            select.select([fd], [], [])
            os.read(fd, 65536)
    if args.watch:
        os.close(fd)
    print(json.dumps(dict(output=str(args.output), tool_calls=sum(counts.values()), short_waits=len(short), alerts=len(alerts))))


if __name__ == '__main__':
    main()
