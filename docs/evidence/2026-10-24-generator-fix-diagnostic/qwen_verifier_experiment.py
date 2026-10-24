"""Fixed generation followed by a yes/no answer check per claim, on the same 16 cases."""
import json
from pathlib import Path

import qwen_fix_experiment as base

VERIFY_SYSTEM = ("You check one claim against one question. Answer true only if the claim states the specific "
                 "information that the question asks for. A related fact that does not give that information is false.")
VERIFY_SCHEMA = {"type": "object", "properties": {"answers_question": {"type": "boolean"}},
                 "required": ["answers_question"]}


def main():
    c = json.load(open(base.CORPUS, encoding='utf-8'))
    P = {p['id']: p for p in c['passages']}
    by_lang = {}
    for p in c['passages']:
        by_lang.setdefault(p['language'], []).append(p['id'])
    picks = [q for q in c['questions'] if q['language'] == 'en'][:4] + \
            [q for q in c['questions'] if q['language'] == 'fr'][:3] + \
            [q for q in c['questions'] if q['language'] == 'nl'][:3]
    cases = []
    for kind, qs, key in (('answerable', picks, 'relevant'), ('unsupported', c['unsupported_questions'][:6], 'near')):
        for q in qs:
            ids = list(q.get(key, []))[:3]
            for pid in by_lang[q['language']]:
                if len(ids) == 3:
                    break
                if pid not in ids:
                    ids.append(pid)
            cases.append((kind, q, [P[i]['text'] for i in ids]))
    rows = []
    for kind, q, sources in cases:
        text, s1 = base.call(base.SYSTEM_NEW, base.prompt_new(q['text'], sources), base.SCHEMA)
        outcome, claims = base.check(text, sources)
        kept, s2 = [], 0.0
        for cl in claims or []:
            v, t = base.call(VERIFY_SYSTEM, f"Question: {q['text']}\nClaim: {cl['text']}", VERIFY_SCHEMA)
            s2 += t
            try:
                if json.loads(v).get('answers_question') is True:
                    kept.append(cl)
            except ValueError:
                pass
        final = outcome if outcome != 'answered' else ('answered' if kept else 'abstained_by_check')
        hit = None
        if kind == 'answerable' and kept:
            blob = ' '.join(x['text'] + ' ' + x['quote'] for x in kept)
            hit = any(e in blob for e in q.get('expected', []))
        rows.append({'kind': kind, 'id': q['id'], 'generation': outcome, 'final': final, 'expected_found': hit,
                     'seconds': round(s1 + s2, 1)})
        print(kind, q['id'], outcome, '->', final, hit, round(s1 + s2, 1), flush=True)
    Path(__file__).with_name('qwen_verifier_results.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    ans = [r for r in rows if r['kind'] == 'answerable']
    uns = [r for r in rows if r['kind'] == 'unsupported']
    print(f"== verifier: answerable kept {sum(r['final'] == 'answered' for r in ans)}/{len(ans)}, expected "
          f"{sum(bool(r['expected_found']) for r in ans)}/{len(ans)}, unsupported refused "
          f"{sum(r['final'] in ('abstained', 'abstained_by_check') for r in uns)}/{len(uns)}")


if __name__ == '__main__':
    main()
