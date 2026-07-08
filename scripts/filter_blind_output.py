import json
from datasets import load_dataset

data = json.load(open('exp/inference/blindset/qwen3_8b_bm25_blindset_A.json'))

ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Blind-A', split='test')
last_turns = {}
for item in ds:
    last_turns[item['session_id']] = item['conversations'][-1]['turn_number']

filtered = [r for r in data if last_turns.get(r['session_id']) == r['turn_number']]
print(f'Filtered: {len(filtered)} entries')

json.dump(filtered, open('exp/inference/blindset/prediction.json', 'w'), ensure_ascii=False)
print('Done')
