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


class StopStringFilter:
    """Removes stop strings from streamed text.
    
    The engine stops on the token that completes a stop string, but earlier tokens may 
    already hold its start ("\\n" then "\\n" for "\\n\\n"). So push() holds back any tail
    that could still grow into a stop string, and cuts the text at the first full match
    """

    def __init__(self, stop):
        self.stop = [s for s in stop if s]      # an empty string would match everywhere
        self.held = ""
        self.stopped = False

    def push(self, text):
        """Adds newly decoded text, returns the part that is safe to send"""
        if self.stopped:
            return ""
        self.held += text
        hits = [i for i in (self.held.find(s) for s in self.stop) if i != -1]
        if hits:
            self.stopped = True
            out, self.held = self.held[:min(hits)], ""
            return out
        keep = self._partial_tail()
        out = self.held[:len(self.held) - keep]
        self.held = self.held[len(self.held) - keep:]
        return out

    def flush(self):
        """At the end of a request: returns held text that never became a stop string."""
        out, self.held = ("" if self.stopped else self.held), ""
        return out

    def _partial_tail(self):
        """Length of the longest tail of held that is a proper prefix of some stop string"""
        best = 0
        for s in self.stop:
            for n in range(min(len(s) - 1, len(self.held)), best, -1):
                if self.held.endswith(s[:n]):
                    best = n
                    break
        return best