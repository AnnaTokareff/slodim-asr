import os
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
from jiwer import wer, cer
from dataset_builder import DatasetBuilder


class Wav2Vec2FineTuner:
    def __init__(self, model_name, audio_dir, output_dir, build_dataset=True, train_csv_path=None, test_csv_path=None):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.output_dir = output_dir
        self.build_dataset = build_dataset
        
        if self.build_dataset == True:
            self.dataset_builder = DatasetBuilder(
            eaf_dir=eaf_dir,
            wav_dir=os.path.join(audio_dir, "wav"),
            seg_dir=os.path.join(audio_dir, "seg"),
            output_dir=output_dir
        )
            self.dataset_builder.build_dataset()
            self.train_csv_path = os.path.join(self.output_dir, "train.csv")
            self.test_csv_path = os.path.join(self.output_dir, "test.csv")

    def _prepare_sample(self, example):
        '''
        prepare audio as required
        '''
        input_values = self.processor(example["path"], sampling_rate=16_000, return_tensors="pt").input_values[0]
        with self.processor.as_target_processor():
            labels = self.processor(example["text"]).input_ids
        return {"input_values": input_values, "labels": labels}

    def _load_and_preprocess_data(self, csv_path):
        '''
        map segmented audio to preprocessing function
        '''
        dataset = load_from_disk(csv_path)
        dataset = dataset.map(
            self._prepare_sample,
            remove_columns=dataset.column_names,
            num_proc=4,
        )
        return dataset

    def get_train_dataset(self):
        train_dataset = self._load_and_preprocess_data(self.train_csv_path)
        return train_dataset

    def get_test_dataset(self):
        test_dataset = self._load_and_preprocess_data(self.test_csv_path)
        return test_dataset

    def _compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer(label_str, pred_str)
        cer = cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}

    def fine_tune(self):
        train_dataset = self.get_train_dataset()
        test_dataset = self.get_test_dataset()
        print("Dataset loaded for training")

        training_args = TrainingArguments(
            output_dir='./output/wav2vec2_results',
            num_train_epochs=30,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
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
            data_collator=lambda data: self.processor.collate(data, return_tensors="pt"),
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)


if __name__ == "__main__":
    # Define directories and paths
    eaf_dir = "./eaf"
    audio_dir = "./audio"
    output_dir = "./output"
    train_csv_path = os.path.join(output_dir, "train.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")
    
    # Instantiate and run Wav2Vec2FineTuner
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    fine_tuner = Wav2Vec2FineTuner(model_name, audio_dir, output_dir, build_dataset=False, 
                                   train_csv_path=train_csv_path, test_csv_path=test_csv_path)
    fine_tuner.fine_tune()

    # Save the finetuned model
    fine_tuner.save_model()
