import os
from typing import List
import struct
import argparse
from sentencepiece import SentencePieceProcessor

MODEL_PATH = "./tokenizer.model"

class Tokenizer:
    def __init__(self, model= None):
        self.model_path = model if model else MODEL_PATH
        assert os.path.isfile(self.model_path), f"Model path {self.model_path} does not exist...!"

        self.sp_model = SentencePieceProcessor(model_file=self.model_path)
        self.vocab: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.unk_id: int = self.sp_model.unk_id()

        assert self.vocab == self.sp_model.GetPieceSize(), "Mismatch in vocabulary size...!"

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
    
    def convert_to_binary(self):
        tokens , scores  = [], []
        for i in range(self.vocab):
            token = self.sp_model.IdToPiece(i)
            score = self.sp_model.GetScore(i)

            if i == self.bos_id:
                token = "\n<bos>\n"
            if i == self.eos_id:
                token = "\n<eos>\n"

            token = token.replace('▁', ' ')
            btoken = token.encode("utf-8")
            
            tokens.append(btoken)
            scores.append(score)

        max_token_len = max(len(token) for token in tokens)
        tokenizer_bin = self.model_path.replace(".model", ".bin")
        with open(tokenizer_bin, "wb") as file:
            file.write(struct.pack("I", max_token_len))
            file.write(struct.pack("I", self.vocab))
            for bytes, score in zip(tokens, scores):
                file.write(struct.pack("fI", score, len(bytes)))
                file.write(bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the SentencePiece model")
    args = parser.parse_args()
    tokenizer = Tokenizer(args.model)
    tokenizer.convert_to_binary()