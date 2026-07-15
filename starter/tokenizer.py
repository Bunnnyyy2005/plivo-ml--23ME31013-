import json
import os
import collections

class BPETokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def train(self, text, vocab_size):
        tokens = list(text.encode("utf-8"))
        for i in range(256, vocab_size):
            counts = collections.Counter(zip(tokens[:-1], tokens[1:]))
            if not counts:
                break
            best = max(counts, key=counts.get)
            self.merges[best] = i
            self.vocab[i] = self.vocab[best[0]] + self.vocab[best[1]]
            
            # Fast BPE replacement using native C-backend string operations
            s = "".join(chr(x) for x in tokens)
            s = s.replace(chr(best[0]) + chr(best[1]), chr(i))
            tokens = [ord(c) for c in s]
            
            self.vocab_size = i + 1

    def encode(self, text):
        b = text.encode("utf-8")
        if not self.merges:
            return list(b)
        
        s = "".join(chr(x) for x in b)
        for pair, new_id in self.merges.items():
            s = s.replace(chr(pair[0]) + chr(pair[1]), chr(new_id))
        return [ord(c) for c in s]

    def decode(self, ids):
        b = b"".join(self.vocab.get(i, b"") for i in ids)
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, "w") as f:
            json.dump({"merges": merges_str}, f)

    def load_from_file(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for k, v in data["merges"].items():
            p1, p2 = map(int, k.split(","))
            self.merges[(p1, p2)] = v
            self.vocab[v] = self.vocab[p1] + self.vocab[p2]
        self.vocab_size = 256 + len(self.merges)

def load(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    tok = BPETokenizer()
    if os.path.exists(path):
        tok.load_from_file(path)
    return tok
