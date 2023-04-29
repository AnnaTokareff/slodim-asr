import json
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder

# referring to https://huggingface.co/blog/wav2vec2-with-ngram
# Install KenLM library
# $ sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
# $ wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
# $ mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
# $ kenlm/build/bin/lmplz -o 5 <"text.txt" > "5gram.arpa"

# add </s> to .arpa:
with open("5gram.arpa", "r") as read_file, open(
    "5gram_correct.arpa", "w"
) as write_file:
    has_added_eos = False
    for line in read_file:
        if not has_added_eos and "ngram 1=" in line:
            count = line.strip().split("=")[-1]
            write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
        elif not has_added_eos and "<s>" in line:
            write_file.write(line)
            write_file.write(line.replace("<s>", "</s>"))
            has_added_eos = True
        else:
            write_file.write(line)

# Load the checkpoint
checkpoint_dir = "./output/checkpoint-8000"
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Extract the vocab of tokenizer
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {
    k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])
}

# Build the ctc decoder
decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path="5gram_correct.arpa",
)

# Wrap the processor with LM
processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder,
)

# Save the processor with LM to the finetuned wav2vec
processor_with_lm.save_pretrained("output/checkpoint-8000")
