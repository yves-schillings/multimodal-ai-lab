"""Evaluate explicitly fictional French/Dutch fixtures with the actual speech adapter.

Generation requires the separate eSpeak NG executable. It is not installed in
the application image. This synthetic integration measurement does not establish
quality for human statements, dialects, background noise or recording devices.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.evaluation import evaluate_transcripts, transcript_metrics
from lab.speech import SpeechEngine


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=Path,default=Path('tests/fixtures/speech_fr_nl.json'))
    ap.add_argument('--audio-dir',type=Path,required=True)
    ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--generate',action='store_true',help='Generate fictional clips with an installed eSpeak NG executable')
    args=ap.parse_args()
    corpus=json.loads(args.corpus.read_text(encoding='utf-8'))
    samples=corpus['samples']
    if len(samples)!=20 or len({x['id'] for x in samples})!=20:
        raise ValueError('Expected twenty unique fictional samples')
    if any(x['language'] not in ('fr','nl') or not x['reference'].strip() for x in samples):
        raise ValueError('French/Dutch nonempty reference labels required')
    if any(sum(x['language']==lang for x in samples)!=10 for lang in ('fr','nl')):
        raise ValueError('Expected ten clips per language')
    args.audio_dir.mkdir(parents=True,exist_ok=True)
    generator=None
    if args.generate:
        generator=subprocess.check_output(['espeak-ng','--version'],text=True).strip()
        for sample in samples:
            output=args.audio_dir/(sample['id']+'.wav')
            subprocess.run(['espeak-ng','-v',sample['language'],'-s',str(sample['speed']),
                '-p',str(sample['pitch']),'-w',str(output),'--',sample['reference']],check=True)
    engine=SpeechEngine(model_dir=args.model_dir)
    observations=[]
    for sample in samples:
        audio=args.audio_dir/(sample['id']+'.wav')
        result=engine.transcribe(audio,language=sample['language'])
        hypothesis=' '.join(seg['text'] for seg in result['segments'])
        metrics=transcript_metrics(sample['reference'],hypothesis)
        observations.append({**sample,'hypothesis':hypothesis,'metrics':metrics,
            'audio_sha256':sha(audio),'duration_seconds':result['duration_seconds'],
            'processing_seconds':result['processing_seconds'],'segment_count':len(result['segments'])})
        print(json.dumps({'sample':sample['id'],'wer':metrics['wer'],'duration_seconds':result['duration_seconds']}),flush=True)
    totals=evaluate_transcripts(observations)
    for lang,counts in totals['by_language'].items():
        rows=[x for x in observations if x['language']==lang]
        counts['audio_seconds']=round(sum(x['duration_seconds'] for x in rows),3)
        counts['processing_seconds']=round(sum(x['processing_seconds'] for x in rows),3)
        counts['real_time_factor']=round(counts['processing_seconds']/counts['audio_seconds'],4)
    report={'date_utc':datetime.now(timezone.utc).isoformat(),'scope':corpus['scope'],
        'representative_human_speech_quality_claim':False,
        'corpus_sha256':sha(args.corpus),'generator':generator,
        'runtime':{'faster_whisper':importlib.metadata.version('faster-whisper'),
            'pyav':importlib.metadata.version('av'),'device':'cpu','precision':'int8',
            'model':'Whisper base','model_file_hashes':{p.name:sha(p) for p in args.model_dir.iterdir() if p.is_file()},
            'vad_filter':True,'beam_size':5,'explicit_language':True},
        'network_disabled_for_run':os.environ.get('LAB_EVALUATION_NETWORK_DISABLED')=='1',
        'aggregation':'Sum substitutions, deletions and insertions divided by total reference words',
        'normalisation':'NFKC, casefold, punctuation ignored; accents and apostrophes retained',
        'results':totals,'samples':observations}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(totals,indent=2),flush=True)


if __name__=='__main__':main()
