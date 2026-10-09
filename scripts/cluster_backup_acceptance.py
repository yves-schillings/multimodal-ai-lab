"""Private recovery exercise: restore copies, never overwrite accepted runtime volumes."""
import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.deploy_local_auth import protect_file


def oc(*args,timeout=240):
    r=subprocess.run(['oc',*args],capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError('Recovery cluster operation failed; private output withheld')
    return r.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir',type=Path,required=True)
    args=parser.parse_args();target=args.private_dir.resolve()
    if target==ROOT or ROOT in target.parents:raise ValueError('Recovery copies must stay outside Git')
    target.mkdir(parents=True,exist_ok=True);run=target/('recovery-'+uuid.uuid4().hex[:12]);run.mkdir()
    report={'passed':False,'checks':[],'scope':'SQLite/files restored offline; MLflow and vector PostgreSQL dumps restored to temporary databases; no live volume replacement','created_at':time.time()}
    def check(name,value):
        report['checks'].append({'name':name,'passed':bool(value)})
        print(('PASS ' if value else 'FAIL ')+name,flush=True)
        if not value:raise RuntimeError('Recovery acceptance failed: '+name)
    def save(name,data):
        path=run/name;path.write_bytes(data);protect_file(path);return path
    ns='multimodal-ai-lab'
    before=json.loads(oc('get','pvc','-n',ns,'-o','json'))
    before={x['metadata']['name']:x['metadata']['uid'] for x in before['items']}
    pods=json.loads(oc('get','pods','-n',ns,'-l','app.kubernetes.io/name=lab-app','-o','json'))['items']
    app=next(x['metadata']['name'] for x in pods if x['status']['phase']=='Running')
    staging='/tmp/lab-recovery-'+uuid.uuid4().hex
    try:
        # Each SQLite backup is a consistent database snapshot; files are copied while no exercise job is running.
        code="""import hashlib,json,shutil,sqlite3,sys
from pathlib import Path
root=Path('/app/data');out=Path(sys.argv[1]);out.mkdir()
for name in ('statements.sqlite3','identity.sqlite3'):
 with sqlite3.connect(root/name) as source,sqlite3.connect(out/name) as dest:source.backup(dest)
for name in ('uploads','models'):
 if (root/name).exists():shutil.copytree(root/name,out/name)
manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest))
"""
        oc('exec','-n',ns,app,'--','python','-c',code,staging)
        archive=save('application-private.tar',oc('exec','-n',ns,app,'--','tar','-C',staging,'-cf','-','.'))
        restored=run/'restored-application';restored.mkdir()
        with tarfile.open(archive) as tar:tar.extractall(restored,filter='data')
        for path in restored.rglob('*'):
            if path.is_file():protect_file(path)
        manifest=json.loads((restored/'manifest.json').read_text())
        check('all restored originals and model artifacts match backup hashes',all(hashlib.sha256((restored/name).read_bytes()).hexdigest()==value for name,value in manifest.items()))
        for name in ('statements.sqlite3','identity.sqlite3'):
            with sqlite3.connect(restored/name) as db:
                check(name+' restored integrity',db.execute('PRAGMA integrity_check').fetchone()[0]=='ok')
                counts={row[0]:db.execute('SELECT COUNT(*) FROM "'+row[0]+'"').fetchone()[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            report[name+'_counts']=counts
        for workload,label,database in [('mlflow','lab-postgres','mlflow'),('vector','lab-vector','lab_vectors')]:
            pods=json.loads(oc('get','pods','-n',ns,'-l','app.kubernetes.io/name='+label,'-o','json'))['items']
            pod=next(x['metadata']['name'] for x in pods if x['status']['phase']=='Running')
            temp='lab_restore_'+uuid.uuid4().hex[:16]
            # Password uses the existing process environment inside its owning database pod; none printed.
            dump=oc('exec','-n',ns,pod,'--','bash','-c','export PGPASSWORD="$POSTGRES_PASSWORD"; exec pg_dump -U "$POSTGRES_USER" -d "$1" -Fc --no-owner --no-acl','backup',database)
            dump_path=save(workload+'-private.dump',dump)
            try:
                oc('exec','-n',ns,pod,'--','bash','-c','export PGPASSWORD="$POSTGRES_PASSWORD"; exec createdb -U "$POSTGRES_USER" "$1"','restore',temp)
                r=subprocess.run(['oc','exec','-i','-n',ns,pod,'--','bash','-c','export PGPASSWORD="$POSTGRES_PASSWORD"; exec pg_restore -U "$POSTGRES_USER" -d "$1" --no-owner --no-acl --exit-on-error','restore',temp],input=dump,capture_output=True,timeout=240)
                check(workload+' PostgreSQL archive restores to isolated database',r.returncode==0)
                sql="SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema');"
                out=oc('exec','-n',ns,pod,'--','bash','-c','export PGPASSWORD="$POSTGRES_PASSWORD"; exec psql -U "$POSTGRES_USER" -d "$1" -Atc "$2"','restore',temp,sql)
                check(workload+' restored database has application tables',int(out.strip())>0)
                sql='SELECT count(*) FROM '+('public.runs' if workload=='mlflow' else 'retrieval.chunks')+';'
                out=oc('exec','-n',ns,pod,'--','bash','-c','export PGPASSWORD="$POSTGRES_PASSWORD"; exec psql -U "$POSTGRES_USER" -d "$1" -Atc "$2"','restore',temp,sql)
                check(workload+' restored database retains actual runtime records',int(out.strip())>0)
                report[workload+'_dump_sha256']=hashlib.sha256(dump).hexdigest()
            finally:
                oc('exec','-n',ns,pod,'--','bash','-c','export PGPASSWORD="$POSTGRES_PASSWORD"; exec dropdb --if-exists -U "$POSTGRES_USER" "$1"','restore',temp)
        after=json.loads(oc('get','pvc','-n',ns,'-o','json'))['items']
        check('all original PVC identities retained',before=={x['metadata']['name']:x['metadata']['uid'] for x in after})
        report['passed']=True
    finally:
        oc('exec','-n',ns,app,'--','python','-c',"from pathlib import Path;import shutil,sys;p=Path(sys.argv[1]);assert p.parent==Path('/tmp') and p.name.startswith('lab-recovery-');shutil.rmtree(p,ignore_errors=True)",staging)
        save('recovery-acceptance.json',json.dumps(report,indent=2).encode())
        print('Private recovery record retained in '+str(run))


if __name__=='__main__':main()
