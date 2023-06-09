import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import numpy as np
import json
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import jiwer
from torch.utils.data import Dataset

torch.cuda.empty_cache()
device_ids = [0, 1, 2, 3]
net = torch.nn.DataParallel(net, device_ids=device_ids)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'

class CustomWhisperDataset(Dataset):
    def __init__(self, data, processor, sampling_rate):
        self._data = [item for item in data if not (isinstance(item["transcription"], float) and np.isnan(item["transcription"]))]
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __getitem__(self, idx):
        audio = self._data[idx]

        input_features = self.processor.feature_extractor(audio["waveform"], sampling_rate=self.sampling_rate).input_features[0]
        labels = self.processor.tokenizer(audio["transcription"]).input_ids
        return {"input_features": input_features, "labels": labels}

    def __len__(self):
        return len(self._data)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class WhisperFineTuner:
    def __init__(self, model_size, output_dir, train_path, test_path, sampling_rate=16000):
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}", language="fr", task="transcribe")
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.train_path = train_path
        self.test_path = test_path

    def get_train_dataset(self):
        with open(self.train_path, "r") as f:
            data = json.load(f)
        dataset = CustomWhisperDataset(data, self.processor, self.sampling_rate)
        return dataset

    def get_test_dataset(self):
        with open(self.test_path, "r") as f:
            data = json.load(f)
        dataset = CustomWhisperDataset(data, self.processor, self.sampling_rate)
        return dataset

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = jiwer.wer(label_str, pred_str)
        cer = jiwer.cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}

    def fine_tune(self):
        train_dataset = self.get_train_dataset()
        test_dataset = self.get_test_dataset()

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=30,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            save_steps=500,
            save_total_limit=5,
            evaluation_strategy="epoch",
            logging_dir="./logs",
            disable_tqdm=True,
            fp16=True,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)

if __name__ == "__main__":
    model_size = "medium"
    sampling_rate = 16000
    train_path = "./output/train.json"
    test_path = "./output/test.json"
    output_dir = "./output/whisper_results"
    fine_tuner = WhisperFineTuner(model_size, output_dir, train_path, test_path, 
                                  sampling_rate=sampling_rate)
    fine_tuner.fine_tune()
    fine_tuner.save_model()
