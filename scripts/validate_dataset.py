#!/usr/bin/env python3
"""Validate released JSONL against byte-preserved source data (standard library only)."""
import hashlib,json,sys,zipfile
from collections import Counter
from pathlib import Path

def validate(root):
    expected={'knowledge_overwriting_faitheval':598,'knowledge_overwriting_conflictqa':1356,'semantic_drift_fineweb':8980,'semantic_drift_faitheval':11960}
    manifest=json.loads((root/'audit/source_manifest.json').read_text())
    assert len(manifest)==3001
    total=0
    with zipfile.ZipFile(root/'original/supplementary_original.zip') as z:
        assert len(z.namelist())==len(manifest)
        for item in manifest:
            raw=z.read(item['path'])
            assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],item['path']
        for config,count in expected.items():
            rows=[json.loads(line) for line in (root/'data'/f'{config}.jsonl').read_text().splitlines()]
            assert len(rows)==count,(config,len(rows))
            assert len({r['example_id'] for r in rows})==count
            counts=Counter()
            for row in rows:
                d=json.loads(z.read(row['source_file']));assert row['context']==d['context']
                counts[row['source_file']]+=1
                if config.startswith('semantic_drift'):
                    index=int(row['example_id'].rsplit('/',1)[1])-1;q=d['qas'][index]
                    assert row['qa_id']==q.get('id','')
                    for key in ['type','probe','question','answer','evidence_span']:assert row[key]==q.get(key,''),(row['example_id'],key)
                else:
                    assert row['question']==d['questions']
                    if config.endswith('conflictqa'):
                        assert row['contextual_answer']==d['choices'][0]
                        assert row['source_choices']==d['choices']
                        assert row['parametric_answer']==d['answer']
                        assert row['justification_of_parametric_memory']==d['justification_of_parametric_memory']
                    else:
                        for key in ['answer','choices','justification']:assert row[key]==d[key]
                        assert row['answer_key']==d['answer key'] and row['source_id']==d['id']
            assert set(counts.values())==({20} if config.startswith('semantic_drift') else {1})
            total+=len(rows)
            print(f'PASS {config}: {len(rows):,} QA rows; {len(counts):,} contexts')
    assert total==22894
    print('PASS 3,001 source hashes; all field mappings and QA strings preserved.')

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python validate_dataset.py /path/to/dataset')
    validate(Path(sys.argv[1]))
