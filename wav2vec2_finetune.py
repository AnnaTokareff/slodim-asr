import os
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from torch.utils.data import Dataset
from jiwer import wer, cer
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'


class CustomWav2Vec2Dataset(Dataset):
    def __init__(self, data, processor, sampling_rate):
        self._data = data
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __getitem__(self, idx):
        audio = self._data[idx]

        if isinstance(audio["transcription"], float) and np.isnan(audio["transcription"]):
            print(f"Skipping audio with NaN transcription: {audio['path']}")
            return None

        waveform = audio["waveform"]
        input_values = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.squeeze(0)
        labels = self.processor.tokenizer(audio["transcription"], return_tensors="pt").input_ids.squeeze(0)

        return {"input_values": input_values, "labels": labels}

    def __len__(self):
        return len(self._data)


class Wav2Vec2FineTuner:
    def __init__(self, model_name, output_dir, train_path, test_path, sampling_rate=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.train_path = train_path
        self.test_path = test_path

    def get_train_dataset(self):
        with open(self.train_path, "r") as f:
            data = json.load(f)
        dataset = CustomWav2Vec2Dataset(data, self.processor, self.sampling_rate)
        return dataset

    def get_test_dataset(self):
        with open(self.test_path, "r") as f:
            data = json.load(f)
        dataset = CustomWav2Vec2Dataset(data, self.processor, self.sampling_rate)
        return dataset

    def _compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer(label_str, pred_str)
        cer = cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}
    
    def data_collator(self, samples):
        input_values = [sample["input_values"] for sample in samples if sample is not None]
        labels = [sample["labels"] for sample in samples if sample is not None]
        input_values_dict = [{"input_values": v} for v in input_values]
        labels_dict = [{"input_ids": l} for l in labels]
        input_values_tensor = self.processor.feature_extractor.pad(input_values_dict, return_tensors="pt").input_values
        labels_tensor = self.processor.tokenizer.pad(labels_dict, return_tensors="pt", padding=True).input_ids
        return {"input_values": input_values_tensor, "labels": labels_tensor}

    def fine_tune(self):
        train_dataset = self.get_train_dataset()
        test_dataset = self.get_test_dataset()
        print("Dataset loaded for training")

        training_args = TrainingArguments(
            output_dir='./output/wav2vec2_results',
            num_train_epochs=30,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            save_steps=500,
            save_total_limit=5,
            evaluation_strategy="epoch",
            report_to="none",
            logging_dir="./logs",
            disable_tqdm=True,
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)


if __name__ == "__main__":
    # Define directories and paths
    audio_dir = "./audio"
    output_dir = "./output"
    train_path = os.path.join(output_dir, "train.json")
    test_path = os.path.join(output_dir, "test.json")
    
    # Instantiate and run Wav2Vec2FineTuner
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    fine_tuner = Wav2Vec2FineTuner(model_name, output_dir, train_path, test_path, sampling_rate=16000)
    fine_tuner.fine_tune()

    # Save the finetuned model
    fine_tuner.save_model()
