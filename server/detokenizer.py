class Detokenizer:
    """Turns one requsts token ids into text, one token at a time.
    
    A character can span several tokens (an emoji is 4 bytes), and decoding a partial 
    character gives the replacement character U+FFFD. So push() decodes a short
    window of recent tokens, emits only the text past what it already emitted, and holds
    everything back while the window ends in U+FFFD.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.prefix = 0         # window start: tokens before this are never decoded again
        self.read = 0           # tokens before this have been emitted as text

    def _decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def push(self, token_id):
        """Adds one token; returns the newly complete text"""
        self.ids.append(token_id)
        done_text = self._decode(self.ids[self.prefix:self.read])
        all_text = self._decode(self.ids[self.prefix:])
        if all_text.endswith("\ufffd"):         # unfinished character: wait
            return ""
        new = all_text[len(done_text):]
        self.prefix, self.read = self.read, len(self.ids)
        return new

    def flush(self):
        """At the end of a request: returns any held-back text, broken characters included."""
        done_text = self._decode(self.ids[self.prefix:self.read])
        all_text = self._decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return all_text[len(done_text):]
    