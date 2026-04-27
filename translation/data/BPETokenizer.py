from collections import defaultdict
import re
from tqdm import tqdm


class BPETokenizer:
    """
    Word-pre-tokenized Byte-Pair Encoding tokenizer.

    Usage
    -----
        bpe = BPETokenizer()
        bpe.train(corpus, num_merges=50)
        ids  = bpe.tokenize("hello world")
        text = bpe.detokenize(ids)
    """

    # Splits text into alternating non-whitespace / whitespace runs.
    _SPLIT_PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^A-Za-z\d\s]+|\s+(?!\S)|\s+")

    def __init__(self) -> None:
        # vocab[id] = token string; seeded with all UTF-8 byte values
        self.vocab: list[str] = [bytes([i]).decode("latin-1") for i in range(256)]
        self.vocab.append("<SOS>"); self.sos_id = len(self.vocab) - 1
        self.vocab.append("<EOS>"); self.eos_id = len(self.vocab) - 1
        self.vocab.append("<PAD>"); self.pad_id = len(self.vocab) - 1

        # merges[n] = (id_a, id_b, new_id) in the order they were learned
        self.merges: list[tuple[int, int, int]] = []
        
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _pretokenize(cls, corpus: str) -> dict[tuple[int, ...], int]:
        """
        Split *corpus* into word-chunks; return a frequency dict mapping
        each unique chunk (as a tuple of UTF-8 byte ids) to its count.
        """
        return [word.encode("utf-8") for word in cls._SPLIT_PAT.findall(corpus)]
        

    @staticmethod
    def _pair_freqs(word_freqs: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
        """
        Count every adjacent id-pair across all word-types, weighted by
        word frequency.  O(V * avg_word_len) where V = unique word-types.
        """
        dictionary = defaultdict(int)
        for word, freq in word_freqs.items():
            for pair in zip(word, word[1:]):
                dictionary[pair] += freq
        

        return dictionary
        ##################################
        #  Q14
        ##################################



    @staticmethod
    def _apply_merge(ids: list[int], a: int, b: int, new_id: int) -> list[int]:
        """Replace every (a, b) occurrence in *ids* with *new_id*."""
        
        result = []
        i = 0

        while i < len(ids):
            if i < (len(ids) - 1) and ids[i] == a and ids[i+1] == b:
                result.append(new_id)
                i +=2
            else:
                result.append(ids[i])
                i += 1
        
        return result





        ##################################
        #  Q14
        ##################################



   
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, corpus: str, num_merges: int = 7500) -> None:
        """
        Learn BPE merge rules from a raw text string.

        Parameters
        ----------
        corpus     : raw text (entire training corpus as one string)
        num_merges : maximum number of pair-merge operations to perform
        """

        word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
        for chunk in self._pretokenize(corpus):
            word_freqs[tuple(chunk)] += 1

        for _ in tqdm(range(num_merges), desc="Learning tokenizer"):
            pair_freqs = self._pair_freqs(word_freqs)
            if not pair_freqs:
                break

            # Find most frequent pair
            best = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
            if pair_freqs[best] < 2:
                break
            a, b = best
            
            # Add to vocabulary
            new_id = len(self.vocab)
            self.vocab.append(self.vocab[a] + self.vocab[b])
            
            # Add to merges
            self.merges.append((a, b, new_id))

            # Apply merge to words
            word_freqs = {tuple(self._apply_merge(word, a, b, new_id)):count for word,count in word_freqs.items()}
            

            


    def tokenize(self, text: str) -> list[int]:
        """
        Encode *text* into token ids by replaying the learned merge rules.

        The text is first split into word-chunks (same boundary rules as
        training), then merges are applied independently within each chunk.
        This keeps each working sequence short and avoids allocating one
        giant intermediate array.

        Parameters
        ----------
        text : raw input string

        Returns
        -------
        List of integer token ids.
        """
        result: list[int] = []
        words = self._pretokenize(text)
        for chunk in words:
            ids = list(chunk)
            pairs = set(zip(ids, ids[1:]))
            for a, b, new_id in self.merges:
                if len(ids) < 2:
                    break
                if (a, b) not in pairs:
                    continue
                ids = self._apply_merge(ids, a, b, new_id)
                pairs = set(zip(ids, ids[1:]))
            result.extend(ids)
        return result

    def detokenize(self, ids: list[int]) -> str:
        """
        Decode token ids back to the original string.

        Parameters
        ----------
        ids : token id sequence from tokenize()

        Returns
        -------
        Reconstructed string.
        """
        return [self.vocab[i] for i in ids]

    def __repr__(self) -> str:
        return (
            f"BPE(vocab_size={len(self.vocab)}, "
            f"num_merges={len(self.merges)})"
        )

    def __len__(self) -> int:
        return len(self.vocab)