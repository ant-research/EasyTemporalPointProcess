from dataclasses import dataclass
from typing import Union, Optional

from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import PaddingStrategy, TruncationStrategy


@dataclass
class TPPDataCollator:
    """
    Data collator that will dynamically pad the inputs of event sequences.

    Args:
        tokenizer ([`EventTokenizer`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: EventTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    truncation: Union[bool, str, TruncationStrategy] = False
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=return_tensors,
        )

        return batch
