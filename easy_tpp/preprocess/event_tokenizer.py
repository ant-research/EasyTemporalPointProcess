import copy
from collections import UserDict
from typing import Optional, Union, Dict, Any, List, Mapping

import numpy as np

from easy_tpp.utils import is_torch_available, is_tf_available, logger, TruncationStrategy, PaddingStrategy, \
    TensorType, is_torch_device, requires_backends, is_numpy_array, py_assert


class BatchEncoding(UserDict):
    """
    Holds the output of the [`~event_tokenizer.EventTokenizer.__call__`],
    [`~event_tokenizer.EventTokenizer.encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
            ('input_ids', 'attention_mask', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
        n_sequences (`Optional[int]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(
            self,
            data: Optional[Dict[str, Any]] = None,
            tensor_type: Union[None, str, TensorType] = None,
            prepend_batch_axis: bool = False
    ):
        super().__init__(data)

        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    def keys(self):
        return self.data.keys()

    def values(self):
        return list(self.data.values())

    def items(self):
        return self.data.items()

    def convert_to_tensors(
            self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = is_numpy_array

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return self

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.

        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self


class EventTokenizer:
    """
    Base class for tokenizer event sequences, vendored from huggingface/transformer
    """
    padding_side: str = "right"
    truncation_side: str = "right"
    model_input_names: List[str] = ["time_seqs", "time_delta_seqs", "type_seqs", "seq_non_pad_mask", "attention_mask"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        self.num_event_types = config.num_event_types
        self.pad_token_id = config.pad_token_id

        self.model_max_length = config.max_len

        self.padding_strategy = config.padding_strategy
        self.truncation_strategy = config.truncation_strategy

        # Padding and truncation side are right by default and overridden in subclasses. If specified in the kwargs, it
        # is changed.
        self.padding_side = config.pop("padding_side", self.padding_side)
        self.truncation_side = config.pop("truncation_side", self.truncation_side)
        self.model_input_names = config.pop("model_input_names", self.model_input_names)

    def _get_padding_truncation_strategies(
            self, padding=False, truncation=None, max_length=None, verbose=False, **kwargs
    ):
        padding_strategy, truncation_strategy = None, None
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            if verbose:
                logger.warning(
                    "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                    " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                    " 'longest_first' truncation strategy"
                )
            truncation = "longest_first"

        # Get padding strategy
        if padding is False:
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                            truncation is None or truncation is False or truncation == "do_not_truncate"
                    ):
                        logger.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`."
                        )
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is not None and truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                max_length = self.model_max_length
            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (not self.pad_token_id):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def _truncate(self,
                  encoded_inputs: Union[Dict[str, Any],
                                        Dict[str, List]],
                  truncation_strategy: TruncationStrategy,
                  truncation_side: str,
                  max_length: Optional[int] = None):
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
            py_assert(max_length is not None, ValueError, 'must pass max_length when truncation is activated!')
            for k, v in encoded_inputs.items():
                seq_ = [seq[:max_length] for seq in v] if truncation_side == 'right' \
                    else [seq[-max_length:] for seq in v]
                encoded_inputs[k] = seq_

        return encoded_inputs

    def pad(
            self,
            encoded_inputs: Union[
                Dict[str, Any],
                Dict[str, List],
            ],
            padding: Union[bool, str, PaddingStrategy] = True,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            verbose: bool = False,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`).

        Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        text followed by a call to the `pad` method to get a padded encoding.

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`]:
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `time_seqs`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        padding_strategy, truncation_strategy, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, truncation=truncation, verbose=verbose
        )

        encoded_inputs = self._truncate(encoded_inputs,
                                        truncation_strategy=truncation_strategy,
                                        max_length=max_length,
                                        truncation_side=self.truncation_side)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_output = self._pad(
            encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            return_attention_mask=return_attention_mask,
        )

        return BatchEncoding(batch_output, tensor_type=return_tensors)

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, Any], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        # check whether we need to pad it
        seq_lens = np.array([len(seq) for seq in required_input])
        is_all_seq_equal_max_length = np.all(seq_lens == max_length)
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and ~is_all_seq_equal_max_length

        batch_output = dict()

        if needs_to_be_padded:
            # time seqs
            batch_output[self.model_input_names[0]] = self.make_pad_sequence(encoded_inputs[self.model_input_names[0]],
                                                                             self.pad_token_id,
                                                                             padding_side=self.padding_side,
                                                                             max_len=max_length)
            # time_delta seqs
            batch_output[self.model_input_names[1]] = self.make_pad_sequence(encoded_inputs[self.model_input_names[1]],
                                                                             self.pad_token_id,
                                                                             padding_side=self.padding_side,
                                                                             max_len=max_length)
            # type_seqs
            batch_output[self.model_input_names[2]] = self.make_pad_sequence(encoded_inputs[self.model_input_names[2]],
                                                                             self.pad_token_id,
                                                                             padding_side=self.padding_side,
                                                                             max_len=max_length,
                                                                             dtype=np.int64)
        else:
            batch_output[self.model_input_names[0]] = np.array(encoded_inputs[self.model_input_names[0]])
            batch_output[self.model_input_names[1]] = np.array(encoded_inputs[self.model_input_names[1]])
            batch_output[self.model_input_names[2]] = np.array(encoded_inputs[self.model_input_names[2]])

        # non_pad_mask; replaced the use of event types by using the original sequence length
        seq_pad_mask = np.full_like(batch_output[self.model_input_names[2]], fill_value=True, dtype=bool)
        for i, seq_len in enumerate(seq_lens):
            seq_pad_mask[i, seq_len:] = False
        batch_output[self.model_input_names[3]] = seq_pad_mask

        if return_attention_mask:
            # attention_mask
            batch_output[self.model_input_names[4]] = self.make_attn_mask_for_pad_sequence(
                batch_output[self.model_input_names[2]],
                self.pad_token_id)
        else:
            batch_output[self.model_input_names[4]] = []

        return batch_output

    @staticmethod
    def make_pad_sequence(seqs,
                          pad_token_id,
                          padding_side,
                          max_len,
                          dtype=np.float32,
                          group_by_event_types=False):
        """Pad the sequence batch-wise.

        Args:
            seqs (list): list of sequences with variational length
            pad_token_id (int, float): optional, a value that used to pad the sequences. If None, then the pad index
            is set to be the event_num_with_pad
            max_len (int): optional, the maximum length of the sequence after padding. If None, then the
            length is set to be the max length of all input sequences.
            pad_at_end (bool): optional, whether to pad the sequnce at the end. If False,
            the sequence is pad at the beginning

        Returns:
            a numpy array of padded sequence


        Example:
        ```python
        seqs = [[0, 1], [3, 4, 5]]
        pad_sequence(seqs, 100)
        >>> [[0, 1, 100], [3, 4, 5]]

        pad_sequence(seqs, 100, max_len=5)
        >>> [[0, 1, 100, 100, 100], [3, 4, 5, 100, 100]]
        ```

        """
        if not group_by_event_types:
            if padding_side == "right":
                pad_seq = np.array([seq + [pad_token_id] * (max_len - len(seq)) for seq in seqs], dtype=dtype)
            else:
                pad_seq = np.array([[pad_token_id] * (max_len - len(seq)) + seq for seq in seqs], dtype=dtype)
        else:
            pad_seq = []
            for seq in seqs:
                if padding_side == "right":
                    pad_seq.append(np.array([s + [pad_token_id] * (max_len - len(s)) for s in seq], dtype=dtype))
                else:
                    pad_seq.append(np.array([[pad_token_id] * (max_len - len(s)) + s for s in seqs], dtype=dtype))

            pad_seq = np.array(pad_seq)
        return pad_seq

    def make_attn_mask_for_pad_sequence(self, pad_seqs, pad_token_id):
        """Make the attention masks for the sequence.

        Args:
            pad_seqs (tensor): list of sequences that have been padded with fixed length
            pad_token_id (int): optional, a value that used to pad the sequences. If None, then the pad index
            is set to be the event_num_with_pad

        Returns:
            np.array: a bool matrix of the same size of input, denoting the masks of the
            sequence (True: non mask, False: mask)


        Example:
        ```python
        seqs = [[ 1,  6,  0,  7, 12, 12],
        [ 1,  0,  5,  1, 10,  9]]
        make_attn_mask_for_pad_sequence(seqs, pad_index=12)
        >>>
            batch_non_pad_mask
            ([[ True,  True,  True,  True, False, False],
            [ True,  True,  True,  True,  True,  True]])
            attention_mask
            [[[ False  True  True  True  True  True]
              [False  False  True  True  True  True]
              [False False  False  True  True  True]
              [False False False  False  True  True]
              [False False False False  True  True]
              [False False False False  True  True]]

             [[False  True  True  True  True  True]
              [False  False  True  True  True  True]
              [False False  False  True  True  True]
              [False False False  False  True  True]
              [False False False False  False  True]
              [False False False False False  False]]]
        ```


        """

        seq_num, seq_len = pad_seqs.shape

        # [batch_size, seq_len]
        seq_pad_mask = pad_seqs == pad_token_id

        # [batch_size, seq_len, seq_len]
        attention_key_pad_mask = np.tile(seq_pad_mask[:, None, :], (1, seq_len, 1))
        subsequent_mask = np.tile(np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)[None, :, :], (seq_num, 1, 1))

        attention_mask = subsequent_mask | attention_key_pad_mask

        return attention_mask

    def make_type_mask_for_pad_sequence(self, pad_seqs):
        """Make the type mask.

        Args:
            pad_seqs (tensor): a list of sequence events with equal length (i.e., padded sequence)

        Returns:
            np.array: a 3-dim matrix, where the last dim (one-hot vector) indicates the type of event

        """
        type_mask = np.zeros([*pad_seqs.shape, self.num_event_types], dtype=np.int32)
        for i in range(self.num_event_types):
            type_mask[:, :, i] = pad_seqs == i

        return type_mask
