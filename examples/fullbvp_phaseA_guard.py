#!/usr/bin/env python3
"""Event-driven dedicated-rollout audit, including code-mode custom tool calls."""
from pathlib import Path
import sys,json,time,ctypes,os,re,datetime
rollout=Path(sys.argv[1]);out=Path(sys.argv[2]);offset=0;short=[];calls=0;waits=0;poll=[];shells=0;missing=[]
libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(0)
if fd<0:raise OSError(ctypes.get_errno(),'inotify_init')
libc.inotify_add_watch(fd,str(rollout).encode(),2)
def audit():
 global offset,calls,waits,shells
 with rollout.open() as f:
  f.seek(offset)
  for line in f:
   try:d=json.loads(line)
   except ValueError:continue
   p=d.get('payload',{});kind=p.get('type','');name=p.get('name','')
   if kind not in ['function_call','custom_tool_call']:continue
   args=p.get('arguments',p.get('input',''));calls+=1
   stamp=datetime.datetime.fromisoformat(d['timestamp'].replace('Z','+00:00')).timestamp()
   if name in ['wait','write_stdin']:
    waits+=1
    try:a=json.loads(args)
    except ValueError:a={}
    if a.get('yield_time_ms',10000)<1800000:short.append(stamp)
   if kind=='custom_tool_call':
    nested=len(re.findall(r'tools\.(?:exec_command|write_stdin)\s*\(',args));shells+=nested
    values=[int(x) for x in re.findall(r'yield_time_ms\s*:\s*(\d+)',args)]
    if nested and len(values)<nested:missing.append(dict(time=d['timestamp'],nested=nested,explicit_yields=len(values)))
    if nested and any(x<1800000 for x in values):short.append(stamp)
    for raw in re.findall(r'cmd\s*:\s*("(?:[^"\\]|\\.)*")',args):
     try:cmd=json.loads(raw)
     except ValueError:continue
     if re.search(r'(?m)^\s*(?:tail\s+-|ps\s+|nvidia-smi(?:\s|$))',cmd):poll.append(dict(time=d['timestamp'],command=cmd[:150]))
  offset=f.tell()
 recent=[x for x in short if x>time.time()-1800]
 out.write_text(json.dumps(dict(rollout=str(rollout),tool_calls=calls,shell_or_nested_wait_calls=shells,wait_calls=waits,
  short_waits_30min=len(recent),short_yields_total=len(short),missing_explicit_shell_yields=missing,
  process_poll_candidates=poll,alarm=len(recent)>5 or len(poll)>2,
  mode='inotify blocking; dedicated thread only; function_call and custom_tool_call parsed; tool-side yield cap documented'),indent=2)+'\n')
audit()
while True:
 os.read(fd,65536);audit()
