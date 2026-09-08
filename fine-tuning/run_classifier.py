"""
This script provides an exmaple to wrap UER-py for classification.
"""
import os
import random
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np
import json
import math
import time

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
            #return temp_output, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model. Map to CPU when CUDA is not available.
        map_location = "cpu" if not torch.cuda.is_available() else {"cuda:1": "cuda:0", "cuda:2": "cuda:0", "cuda:3": "cuda:0"}
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=map_location), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    loss_sum, n_seen = 0.0, 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            loss, logits = args.model(src_batch, tgt_batch, seg_batch)
        # The model returns the mean CE loss over the batch (NLLLoss, reduction='mean';
        # a vector under DataParallel — hence .mean()). Weight by batch size to recover
        # the dataset-mean dev loss for best-on-dev selection by `loss` (parity with
        # netFound's eval_loss).
        bsz = tgt_batch.size(0)
        loss_sum += float(loss.mean().item()) * bsz
        n_seen += bsz
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        # Write confusion matrix into configured artifacts directory (falls back to CWD).
        output_dir = Path(getattr(args, "artifacts_output_path", None) or ".")
        output_dir.mkdir(parents=True, exist_ok=True)
        cm_path = output_dir / "confusion_matrix"
        with cm_path.open("w") as f:
            for cf_a in cf_array:
                f.write(str(cf_a) + "\n")
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    avg_loss = loss_sum / n_seen if n_seen > 0 else 0.0
    return correct / len(dataset), confusion, avg_loss


def macro_f1_from_confusion(confusion):
    """Macro-averaged F1 from a [pred, gold] confusion matrix.

    Averaged over all `labels_num` classes (classes absent from the predictions
    contribute F1=0), matching sklearn's f1_score(average="macro") over the full
    label set. Used for best-on-dev model selection so the protocol matches the
    netFound side (which selects on eval_f1_macro)."""
    eps = 1e-9
    n = confusion.size()[0]
    if n == 0:
        return 0.0
    f1_total = 0.0
    for i in range(n):
        tp = confusion[i, i].item()
        p = tp / (confusion[i, :].sum().item() + eps)
        r = tp / (confusion[:, i].sum().item() + eps)
        f1_total += 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
    return f1_total / n


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument(
        "--artifacts_output_path",
        type=str,
        default=None,
        help="Directory where confusion matrix and other artifacts will be stored.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["accuracy", "f1_macro", "loss"],
        default="f1_macro",
        help="Dev-set metric used to select the best epoch's checkpoint (best-on-dev). "
             "'loss' picks the minimum dev cross-entropy (uses the full probability mass "
             "→ lower selection variance than the thresholded f1_macro). Kept consistent "
             "across NTFMs for a fair comparison (default: f1_macro).",
    )

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Opt-in single-run bit-reproducibility: the osfing pipeline sets
    # OSFING_DETERMINISTIC=1 (--param training.deterministic=true); no-op otherwise.
    # use_deterministic_algorithms raises on ops without a deterministic kernel —
    # intentional (a silent fallback would defeat the purpose). Slows training and
    # does not remove cross-node/GPU variance.
    if os.environ.get("OSFING_DETERMINISTIC") == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        print("Deterministic mode ON (torch.use_deterministic_algorithms, cuDNN benchmark off)")

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    
    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    # -inf so the first epoch always saves, regardless of metric sign (in particular the
    # negated dev loss, which can be < -1).
    total_loss, best_result, best_epoch = 0.0, float("-inf"), 0

    # Per-epoch metrics channel for the NTFM-OSfing pipeline. When the parent process sets
    # OSFING_EPOCH_LOG, one record per epoch (plus a final {"kind": "summary"} record) is
    # appended to that file as a YAML sequence item in flow style — `- ` + JSON, which is a
    # YAML subset — with flush + fsync per line, so a killed job keeps every finished epoch.
    # Metric names (train.loss / dev.accuracy / dev.f1_macro / dev.loss, 1-based epoch) match
    # the netFound fork's EpochLogCallback. Dependency-free; no env var -> no-op.
    _epoch_log_path = os.environ.get("OSFING_EPOCH_LOG")

    def _append_epoch_record(record):
        if not _epoch_log_path:
            return
        try:
            clean = {
                k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in record.items()
            }
            with open(_epoch_log_path, "a", encoding="utf-8") as f:
                f.write("- " + json.dumps(clean) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[WARNING] could not append to OSFING_EPOCH_LOG ({_epoch_log_path}): {e}")

    print("Start training.")
    train_t0 = time.time()

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        epoch_t0 = time.time()
        epoch_loss, epoch_steps = 0.0, 0
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            loss_val = loss.item()
            total_loss += loss_val
            epoch_loss += loss_val
            epoch_steps += 1
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        dev_accuracy, dev_confusion, dev_loss = evaluate(args, read_dataset(args, args.dev_path))
        dev_f1_macro = macro_f1_from_confusion(dev_confusion)
        # Higher-is-better selection score. For `loss` we negate the dev loss so the same
        # argmax-style comparison below picks the MINIMUM dev loss.
        if args.selection_metric == "f1_macro":
            selection_score = dev_f1_macro
        elif args.selection_metric == "loss":
            selection_score = -dev_loss
        else:
            selection_score = dev_accuracy

        # Epoch-mean training loss (the "Avg loss" printed every report_steps is a windowed
        # value that straddles epoch boundaries; this is the per-epoch mean).
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        print(
            "Epoch {} done: train.loss={:.4f} dev.accuracy={:.4f} dev.f1_macro={:.4f} "
            "dev.loss={:.4f} ({:.0f} s)".format(
                epoch, avg_loss, dev_accuracy, dev_f1_macro, dev_loss, time.time() - epoch_t0
            )
        )
        _append_epoch_record({
            "epoch": epoch,
            "train.loss": float(avg_loss),
            "dev.accuracy": float(dev_accuracy),
            "dev.f1_macro": float(dev_f1_macro),
            "dev.loss": float(dev_loss),
            "train_steps": int(epoch_steps),
            "elapsed_s": round(time.time() - epoch_t0, 3),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        })

        # Best-on-dev model selection by the configured metric (default f1_macro).
        if selection_score > best_result:
            best_result = selection_score
            best_epoch = epoch
            save_model(model, args.output_model_path)

    # Training summary for the pipeline's run log (the parent re-derives best_epoch from the
    # per-epoch records and can cross-check it against this value). Peak GPU memory is
    # measured HERE, in the process that actually trained.
    summary = {
        "kind": "summary",
        "best_epoch": int(best_epoch),
        "best_result": float(best_result),
        "selection_metric": str(args.selection_metric),
        "epochs_completed": int(args.epochs_num),
        "instances_num": int(instances_num),
        "train_steps": int(args.train_steps),
        "labels_num": int(args.labels_num),
        "device": str(args.device),
        "train_runtime_s": round(time.time() - train_t0, 3),
    }
    try:
        if torch.cuda.is_available():
            summary["gpu.peak_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    _append_epoch_record(summary)

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path), True)


if __name__ == "__main__":
    main()
