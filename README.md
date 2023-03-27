# ASR system for SLODiM platform

## Files

- Whisper:

  - [Transcribing](whisper_transcribe.py)

- Wav2Vec 2.0:

  - [Fine-tuning](wav2vec2_finetune.py)

- Data pre-processing & post-processing

  - [Data preprocessor](data_preprocessor.py)
    - Rename EAF files and audio files
    - Convert audio into WAV format
  - [Dataset builder](dataset_builder.py)
    - Slice audio into segments according to annotations
    - Pair segment filename and transcriptions into CSV files
  - [Audio pre-treatment](audio_pre-treatment.py)
  - [Text normalizer](normalizer.py)

- Archived:
  - [Whisper with Pyannote pipeline](archived/whisper_pyannote.py)
  - [whisper_timestamped](archived/whisper_ts_norm.py)
  - [whisperX](archived/whisperx_norm.py)
  - [Wav2Vec 2.0 with Pyannote pipeline](archived/wav2vec_pyannote.py)

## Usage

### Preprocessing data and building dataset:

```python
python3 data_preprocessor.py
python3 dataset_builder.py
```

Notice that the required file structure is as follow:

- Original audio: `./audio/m4a`
- Original EAF file: `./eaf`

If your folder name is different, you might want to change the corresponding variables at the end of each file.

The code will create the following folders (if not existing) and output result into them:

- WAV audio: `./audio/wav`
- Segmented WAV audio: `./audio/seg`, with individual folders corresponding to each original audio
- CSV file: `./output`, with full dataset as `data.csv`, training set as `train.csv`, and test set (10%) as `test.csv`

### Fine-tuning Wav2Vec 2.0:

```python
python3 wav2vec2_finetune.py
```

Notice that if you have never run dataset_builder before, you need to change instantiation at the end as follow:

```python
if __name__ == "__main__":
	...
    fine_tuner = Wav2Vec2FineTuner(model_name, audio_dir, output_dir, build_dataset=True)
```

## To do list🗓

- [x] Test each model on the whole dataset

- [x] Fix problem with saving aligned data for WhisperX

- [ ] Finetune Wav2Vec on our data (in progress)

- [ ] Test Wav2Vec and get the transcriptions

- [ ] Figure out how to deal with disfluencies (in progress)

- [ ] Find the ways to detect backchannels

## Citations

- OpenAI Whisper

```bibtex
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={GitHub repository},
  publisher={GitHub}
  year={2022}
  howpublished = {\url{https://github.com/openai/whisper}}
}
```

- Whisper_timestamped

```bibtex
@misc{lintoai2023whispertimestamped,
  title={whisper-timestamped},
  author={Louradour, J{\'e}r{\^o}me},
  journal={GitHub repository},
  year={2023},
  publisher={GitHub},
  howpublished = {\url{https://github.com/linto-ai/whisper-timestamped}}
}
```

- WhisperX

```bibtex

@misc{bain2022whisperx,
  author = {Bain, Max and Han, Tengda},
  title = {WhisperX},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/m-bain/whisperX}},
}
```
