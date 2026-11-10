#!/usr/bin/env python3
"""Inotify-only dedicated rollout guard; understands quoted JS object keys."""
import argparse,ctypes,datetime,json,re,time
from pathlib import Path

def code_mask(text):
    """Ignore quoted payload data when counting actual code-mode tool calls."""
    chars=list(text);starts=set();i=0
    while i<len(text):
        if text[i] in ['"',"'",'`']:
            quote=text[i];starts.add(i);chars[i]=' ';i+=1
            while i<len(text):
                ch=text[i];chars[i]=' '
                if ch=='\\':
                    i+=1
                    if i<len(text):chars[i]=' '
                elif ch==quote:i+=1;break
                i+=1
        else:i+=1
    return ''.join(chars),starts

def audit(source,dest):
    calls=0;waits=0;short=[];shells=0;missing=[];poll=[]
    for line in source.read_text().splitlines():
        try:d=json.loads(line)
        except ValueError:continue
        p=d.get('payload',{});kind=p.get('type','');name=p.get('name','')
        if kind not in ['function_call','custom_tool_call']:continue
        calls+=1;args=p.get('arguments',p.get('input',''));stamp=d.get('timestamp','')
        try:epoch=datetime.datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp()
        except ValueError:epoch=0
        if name.rsplit('.',1)[-1] in ['wait','write_stdin']:
            waits+=1
            try:obj=json.loads(args)
            except (ValueError,TypeError):obj={}
            if obj.get('yield_time_ms',10000)<1800000:short.append(epoch)
        if not isinstance(args,str):continue
        masked,strings=code_mask(args)
        visible=lambda match:masked[match.start()]!=' ' or match.start() in strings
        waits+=len(re.findall(r'tools\.(?:wait|write_stdin)\s*\(',masked))
        count=len(re.findall(r'tools\.(?:exec_command|write_stdin)\s*\(',masked));shells+=count
        values=[int(m.group(1)) for m in re.finditer(r'[\"\x27]?yield_time_ms[\"\x27]?\s*:\s*(\d+)',args) if visible(m)]
        if count and len(values)<count:missing.append(dict(time=stamp,calls=count,explicit_yields=len(values)))
        if count and any(v<1800000 for v in values):short.append(epoch)
        for match in re.finditer(r'[\"\x27]?cmd[\"\x27]?\s*:\s*("(?:[^"\\]|\\.)*")',args):
            if not visible(match):continue
            raw=match.group(1)
            try:cmd=json.loads(raw)
            except ValueError:continue
            if re.search(r'(?m)^\s*(?:tail\s+-|ps\s+|nvidia-smi(?:\s|$))',cmd):poll.append(dict(time=stamp,command=cmd[:160]))
    recent=sum(x>time.time()-1800 for x in short)
    result=dict(rollout=str(source),tool_calls=calls,shell_calls=shells,wait_calls=waits,short_yields_total=len(short),short_waits_30min=recent,missing_explicit_shell_yields=missing,process_poll_candidates=poll,alarm=recent>5 or len(poll)>2,mode='inotify blocking; quoted and unquoted JS keys supported; requested yield is recorded, tool-side capping excluded')
    temp=dest.with_suffix(dest.suffix+'.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rollout',type=Path);p.add_argument('out',type=Path);p.add_argument('--once',action='store_true');a=p.parse_args()
    if a.once:print(json.dumps(audit(a.rollout,a.out)))
    else:
        import os
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(0)
        if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        libc.inotify_add_watch(fd,str(a.rollout).encode(),2);audit(a.rollout,a.out)
        while True:os.read(fd,65536);audit(a.rollout,a.out)
