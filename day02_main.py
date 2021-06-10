import argparse
from functools import partial
import os
import time

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.io import BatchSampler
from paddle.io import DistributedBatchSampler
from paddle.metric import Accuracy
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import CrossEntropyLoss
from paddle.optimizer import AdamW
import paddlenlp
from paddlenlp.data import Pad
from paddlenlp.data import Stack
from paddlenlp.data import Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup


def get_data_path(data_dir='data', data_class='lcqmc', split='train'):

    data_path = os.path.join(data_dir, data_class, split + '.tsv')
    return data_path


def create_dataset(data_class, split, is_test=False):
    
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fRead:
            for line in fRead:  
                if is_test:
                    query, title = line.strip('\n').split('\t')
                    yield {'query': query, 'title': title}
                else:
                    query, title, label = line.strip('\n').split('\t')
                    yield {'query': query, 'title': title, 'label': label}

    data_path = get_data_path(data_class=data_class, split=split)

    dataset = load_dataset(read, data_path=data_path, lazy=False)
    return dataset


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example['query'], example['title']

    encoded_inputs = tokenizer(
        text=query,
        text_pair=title,
        max_seq_len=max_seq_length
    )

    input_ids, token_type_ids = encoded_inputs['input_ids'], encoded_inputs['token_type_ids']

    if not is_test:
        label = np.array([example['label']], dtype='int64')
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_data_loader(args, data_class, convert_example, tokenizer):
    
    # 1. train and dev Dataset
    train_dataset = create_dataset(data_class=data_class, split='train')
    dev_dataset = create_dataset(data_class=data_class, split='dev')

    # 2. train and dev Sampler
    train_batch_sampler = DistributedBatchSampler(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    dev_batch_sampler = BatchSampler(
        dataset=dev_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=512)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        Stack(dtype='int64')
    ): [data for data in fn(samples)]

    # 3. train and dev DataLoader
    train_data_loader = DataLoader(
        dataset=train_dataset.map(trans_func),
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
        num_workers=2
    )
    dev_data_loader = DataLoader(
        dataset=dev_dataset.map(trans_func),
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
        num_workers=2
    )

    return train_data_loader, dev_data_loader


def create_infer_data_loader(args, data_class, convert_example, tokenizer):
    
    # 1. test Dataset
    test_dataset = create_dataset(data_class=data_class, split='test', is_test=True)
 
    # 2. test Sampler
    test_batch_sampler = BatchSampler(
        dataset=test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=512, is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    ): [data for data in fn(samples)]

    # 3. train and dev DataLoader
    test_data_loader = DataLoader(
        dataset=test_dataset.map(trans_func),
        batch_sampler=test_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
        num_workers=2
    )

    return test_data_loader
    

class PointwiseMatching(nn.Layer):

    def __init__(self, pretrained_model, drop_rate=0.1):
        super().__init__()

        self.ptm = pretrained_model
        self.dropout = nn.Dropout(drop_rate)

        self.classifier = nn.Linear(in_features=self.ptm.config['hidden_size'], out_features=2)

    def forward(self, 
                input_ids, 
                token_type_ids=None, 
                position_ids=None, 
                attention_mask=None):
        
        _, cls_embedding = self.ptm(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            position_ids=position_ids, 
            attention_mask=attention_mask
        )
        cls_embedding = self.dropout(cls_embedding)

        logits = self.classifier(cls_embedding)
        # probs = F.softmax(logits)

        return logits


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    
    model.eval()

    metric.reset()
    losses = []

    for batch in data_loader:
        input_ids, token_type_ids, labels = batch

        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)

        loss = criterion(probs, labels)
        losses.append(loss.numpy())

        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

    model.train()
    metric.reset()

    return np.mean(losses), acc

def train(args, model, train_data_loader, dev_data_loader, data_class):
    
    num_training_steps = len(train_data_loader) * args.epochs

    # learning_rate scheduler
    lr_scheduler = LinearDecayWithWarmup(
        learning_rate=args.learning_rate, 
        total_steps=num_training_steps, 
        warmup=args.warmup_proportion
    )
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
    ]

    # Optimizer
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params
    )

    # Loss
    criterion = CrossEntropyLoss()

    # Metric
    metric = Accuracy()

    """******************************************* training *****************************************""" 

    global_step = 0
    best_val_acc = 0.0

    save_dir = os.path.join(args.save_dir, data_class)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_param_path = os.path.join(save_dir, 'bets_model_state.pdparams')

    tic_train = time.time()

    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):

            input_ids, token_type_ids, labels = batch
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)

            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1

            if global_step % args.print_every_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"% (
                    global_step, epoch, step, loss, acc, args.print_every_steps / (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.evaluate_every_steps == 0:
                loss, acc = evaluate(model, criterion, metric, dev_data_loader)
                print(f'eval dev loss: {loss:.5f}, acc: {acc:.5f}')
                
                if acc > best_val_acc:
                    best_val_acc = acc

                    print(f'save model at global step {global_step}, best val acc is {best_val_acc:.5f}!')

                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)


@paddle.no_grad()
def predict(model, data_loader):

    batch_probs = []

    model.eval()

    for batch in data_loader:
        input_ids, token_type_ids = paddle.to_tensor(batch[0]), paddle.to_tensor(batch[1])

        batch_prob = model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()

        batch_probs.append(batch_prob)

    batch_probs = np.concatenate(batch_probs, axis=0)

    return batch_probs
        
def do_predict(args, model, data_class, test_data_loader):
    
    y_probs = predict(model, test_data_loader)
    y_preds = np.argmax(y_probs, axis=1)


    test_dataset = create_dataset(data_class, split='test', is_test=True)
    
    with open(data_class + '.tsv', 'w', encoding='utf-8') as fWriter:
        
        fWriter.write('index\tprediction\n')
        for idx, y_pred in enumerate(y_preds):
            fWriter.write('{}\t{}\n'.format(idx, y_pred))
            
            text_pair = test_dataset[idx]
            text_pair['label'] = y_pred
            print(text_pair)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for AdamW."
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.0,
        type=float,
        help="Proportion of training steps to perform linear learning rate warmup for."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--save_dir",
        default='checkpoint',
        type=str,
        help="save model path"
    )
    parser.add_argument(
        "--print_every_steps",
        default=10,
        type=int,
        help="how many steps to print log"
    )
    parser.add_argument(
        "--evaluate_every_steps",
        default=100,
        type=int,
        help="how many steps do evaluate"
    )

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    
    # TODO: 对不同的数据集配置不同的参数
    args = parse_args()
    print(args)
    #paddle.seed(args.seed)

    # data_classes = ['lcqmc', 'bq_corpus', 'paws-x'] 3000多，1500多，700多 batch_size=64
    # paws-x        0.85350 epoch=2, 0.85650 epoch=4, 0.8580 epoch=5 
    # bq_corpus     0.84860 epoch=1, 0.86000 epoch=2  0.86030 epoch=4 0.86040 epoch=5

    data_classes = ['bq_corpus']

    for data_class in data_classes:

        pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
        tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')

        model = PointwiseMatching(pretrained_model=pretrained_model)

        train_data_loader, dev_data_loader = create_data_loader(
            args=args, 
            data_class=data_class, 
            convert_example=convert_example, 
            tokenizer=tokenizer
        )

        train(args, model, train_data_loader, dev_data_loader, data_class)


        pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
        model = PointwiseMatching(pretrained_model)

        save_param_path = os.path.join(args.save_dir, data_class, 'bets_model_state.pdparams')
        state_dict = paddle.load(save_param_path)
        model.set_dict(state_dict)

        test_data_loader = create_infer_data_loader(
            args=args, 
            data_class=data_class, 
            convert_example=convert_example, 
            tokenizer=tokenizer
        )

        do_predict(args, model, data_class, test_data_loader)



