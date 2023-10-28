import argparse
import itertools
import json
import os
import random
import time

import torch
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ds_collections = {
    'flickr': {
        'train': 'data/flickr30k/flickr30k_karpathy_test.json',
        'test': 'data/flickr30k/flickr30k_karpathy_test.json',
    },
    'nocaps': {
        'train': '',
        'test': 'data/nocaps/nocaps_val.json',
    },
}


class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, train, test, tokenizer, prompt, few_shot=0):
        self.images = json.load(open(test))['images']
        self.tokenizer = tokenizer
        self.prompt = prompt

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = json.load(open(train))['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id, image_path = self.images[idx]['id'], self.images[idx][
            'image']

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                few_shot_prompt += self.prompt.format(
                    sample['image']) + f" {sample['caption']}"

        return {
            'image_id':
            image_id,
            'input_tokens':
            self.tokenizer(few_shot_prompt + self.prompt.format(image_path),
                           return_tensors='pt').input_ids
        }


def collate_fn(inputs):

    image_ids = [_['image_id'] for _ in inputs]
    input_tokens = torch.cat([_['input_tokens'] for _ in inputs], dim=0)

    return image_ids, input_tokens


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(torch.distributed.get_rank())

    prompt = '<img>{}</img>Describe the image in English:'

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map='cuda', trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)

    random.seed(args.seed)
    dataset = CaptionDataset(
        train=ds_collections[args.dataset]['train'],
        test=ds_collections[args.dataset]['test'],
        tokenizer=tokenizer,
        prompt=prompt,
        few_shot=args.few_shot,
    )
    coco_karpathy_test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    image_ids = []
    captions = []
    for _, (ids, input_ids) in tqdm(enumerate(coco_karpathy_test_loader)):
        pred = model.generate(
            input_ids=input_ids.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=30,
            min_new_tokens=8,
            length_penalty=0,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
        )
        image_ids.extend(ids)
        captions.extend([
            tokenizer.decode(_[input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ])

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_ids = [None for _ in range(world_size)]
    merged_captions = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_ids, image_ids)
    torch.distributed.all_gather_object(merged_captions, captions)

    merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
    merged_captions = [
        _ for _ in itertools.chain.from_iterable(merged_captions)
    ]

    if torch.distributed.get_rank() == 0:
        results = []
        for image_id, caption in zip(merged_ids, merged_captions):
            results.append({
                'image_id': int(image_id),
                'caption': caption,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{args.dataset}_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

        coco = COCO(ds_collections[args.dataset]['test'])
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()

        print(coco_eval.eval.items())
    torch.distributed.barrier()
