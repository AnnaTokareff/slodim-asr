## ASR system for SLODiM platform

- Whisper:
  - [with Pyannote pipeline](whisper_pyannote.py)
  - [with whisper_timestamped](whisper_ts_norm.py)
  - [with whisperX](whisperx_norm.py)

- Wav2Vec 2.0:
  - [with Pyannote pipeline](wav2vec_pyannote.py)

- [Normalizer](normalizer.py)

<h2 align="left" id="To do list"> To do list🗓</h2>

* [ ] Test each model on the whole dataset

* [ ] Fix problem with saving aligned data for WhisperX

* [ ] Train Wav2Vec on our data

* [ ] Test Wav2Vec and get the transcriptions

* [ ] Figure out how to deal with disfluencies

* [ ] Find the ways to detect backchannels



## Citations

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
