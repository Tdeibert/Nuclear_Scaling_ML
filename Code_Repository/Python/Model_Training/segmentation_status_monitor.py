"""Read-only process/log monitor; publishes an atomic Markdown status file."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--log',type=Path,required=True);ap.add_argument('--pid',type=int,required=True)
    args=ap.parse_args();proc=Path('/proc')/str(args.pid)
    identity=(proc/'stat').read_text().split()[21] if proc.exists() else None
    start=time.time()-(float(Path('/proc/uptime').read_text().split()[0])-int(identity)/os.sysconf('SC_CLK_TCK')) if identity else time.time()
    output=args.run/'SEGMENTATION_STATUS.md'
    while True:
        running=False
        try:
            fields=(proc/'stat').read_text().split()
            running=fields[21]==identity and fields[2]!='Z'
        except FileNotFoundError:pass
        log=args.log.read_text(errors='replace') if args.log.exists() else ''
        clean=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',log).replace('\r','\n')
        planes=re.findall(r't=(\d+) z=(\d+):\s*(\d+) nuclei',clean)
        unique={(int(t),int(z)) for t,z,_ in planes}
        cp={}
        for path in args.run.glob('**/_checkpoint.json'):
            try:
                candidate=json.loads(path.read_text())
                if candidate.get('stage')=='segmentation':cp=candidate;break
            except (ValueError,OSError):pass
        done=cp.get('completed_t',[]);elapsed=time.time()-start
        complete=cp.get('status')=='complete' or 'Segmentation complete:' in clean
        state='COMPLETE' if complete else 'RUNNING' if running else 'STOPPED — check log'
        last=f'T{planes[-1][0]}, Z{planes[-1][1]} — {planes[-1][2]} nuclei' if planes else 'No completed plane reported yet'
        eta=f'{elapsed/len(done)*(10-len(done))/60:.0f} minutes (timepoint-average estimate)' if done and len(done)<10 else 'Pending enough completed timepoints' if not complete else 'Finished'
        bar='█'*len(done)+'░'*(10-len(done))
        age=time.time()-args.log.stat().st_mtime if args.log.exists() else 0
        lines=[x for x in clean.splitlines() if x.strip()][-8:]
        text=f'''# Segmentation status

Updated: {datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S %Z')} (refreshes every 15 seconds)

**{state}** — PID {args.pid}

- Elapsed: {elapsed/60:.1f} minutes
- Checkpointed timepoints: [{bar}] {len(done)}/10
- Saved timepoints: {done}
- Completed planes reported in log: {len(unique)}
- Latest plane: {last}
- Estimated remaining: {eta}
- Last log update: {age:.0f} seconds ago

Timepoints count only after checkpointing. Plane messages may pause during inference or output writes.

Run: `{args.run}`

Full log: `{args.log}`

## Recent output

```text
{chr(10).join(lines)}
```
'''
        temp=output.with_suffix('.md.tmp');temp.write_text(text);os.replace(temp,output)
        if not running:break
        time.sleep(15)


if __name__=='__main__':main()
