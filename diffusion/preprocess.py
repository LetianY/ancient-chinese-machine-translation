import json
import codecs

def convert_to_jsonl(src_file, trg_file, output_file):
    with (open(src_file, 'r', encoding='utf-8') as src,
          open(trg_file, 'r', encoding='utf-8') as trg,
          open(output_file, 'w', encoding='utf-8') as out):
        for src_line, trg_line in zip(src, trg):
            src_line = src_line.strip()
            trg_line = trg_line.strip()

            if not src_line or not trg_line:
                continue

            json_line = json.dumps({'src': src_line, 'trg': trg_line}, ensure_ascii=False)
            out.write(json_line + '\n')


src_file_path = '../data_splited/c_{}.txt'
trg_file_path = '../data_splited/m_{}.txt'
output_file_path = './data/{}.jsonl'

# Convert to JSON Lines
for dataset in ['train', 'test']:
    src_file = src_file_path.format(dataset)
    trg_file = trg_file_path.format(dataset)
    if dataset == 'test':
        output_file = output_file_path.format(dataset)
        convert_to_jsonl(src_file, trg_file, output_file)
    else:
        with codecs.open(src_file, "r", "utf-8") as f:
            source_data = f.readlines()
        with codecs.open(trg_file, "r", "utf-8") as f:
            target_data = f.readlines()

        indices = list(range(len(source_data)))
        val_size = int(len(source_data) * 0.2)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        with open(output_file_path.format('train'), 'w', encoding='utf-8') as out:
            for i in train_indices:
                src_line = source_data[i].strip()
                trg_line = target_data[i].strip()
                if not src_line or not trg_line:
                    continue
                json_line = json.dumps({'src': src_line, 'trg': trg_line}, ensure_ascii=False)
                out.write(json_line + '\n')

        with open(output_file_path.format('valid'), 'w', encoding='utf-8') as out:
            for i in val_indices:
                src_line = source_data[i].strip()
                trg_line = target_data[i].strip()
                if not src_line or not trg_line:
                    continue
                json_line = json.dumps({'src': src_line, 'trg': trg_line}, ensure_ascii=False)
                out.write(json_line + '\n')






