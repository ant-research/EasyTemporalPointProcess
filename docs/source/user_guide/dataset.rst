===========================================
Expected Dataset Format and Data Processing
===========================================

Required format
===================================

In EasyTPP we use the data in Gatech format, i.e., each dataset is a dict containing the following keys as

.. code-block:: bash

    dim_process: 5 # num of event types (no padding)
    'train': [[{'idx_event': 2, 'time_since_last_event': 1.0267814, 'time_since_last_same_event': 1.0267814, 'type_event': 3, 'time_since_start': 1.0267814}, {'idx_event': 3, 'time_since_last_event': 0.4029268, 'time_since_last_same_event': 1.4297082, 'type_event': 0, 'time_since_start': 1.4297082},...,],[{}...{}]]

where `dim_process` refers to the number of event types (without padding) and
`train` (or `dev` / `test`) contains a list of list which corresponds to an event sequence each.

Each pickle file generates a set of event sequences, each containing three sub sequences:

1. `time_seqs`: absolute timestamps of the events, correspond to `time_since_last_event`.
2. `time_delta_seqs`: relative timestamps of the events, correspond to `time_since_last_same_event`.
3. `type_seqs`: types of the events, correspond to `type_event`. Be noted that the event type index `starts from 0`.


Some widely-used open source datasets in Gatech format can be found at `Google Drive <https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w>`_, which are provided by researchers. We use them for validating and benchmarking EasyTPP models.

Data processing
===================================

The data processing follows the similar pipeline as in official code of `AttNHP <https://github.com/yangalan123/anhp-andtt>`_. We name it the process of `event tokenize`.


Sequence padding
----------------


time_seqs, time_delta_seqs and type_seqs are firstly padded to `the max length of the whole dataset` and then fed into the model in batch.

.. code-block:: bash

    input: raw event sequence (e_0, e_1, e_2, e_3) and max_len=6 # the max length among all data seqs

    output:

            index:    0,          1,         2,         3，      4         5
            dtimes:   0，     t_1-t_0,    t_2-t_1,   t_3-t_2， time_pad, time_pad
            types:    e_0,      e_1,        e_2,       e_3，  type_pad, type_pad


By default, we set the value of time_pad and type_pad to be the *num_event_types* (because we assume the event type index starts from 0, therefore the integer value of num_event_types is unused).

Sequence masking
----------------


After padding, we perform the masking for the event sequences and generate three more seqs: batch_non_pad_mask, attention_mask, type_mask：

1. `batch_non_pad_mask`: it indicates the position of masks in the sequence.
2. `attention_mask`: it indicates the masks used in the attention calculation (one event can only attend to its past events).
3. `type_mask`: it uses one-hot vector to represent the event type. The padded event is a zero vector.

Finally, each batch contains six elements: time_seqs, time_delta_seqs, event_seq, batch_non_pad_mask, attention_mask, type_mask. The implementation of padding mechanism can be found at [event_toknizer.py](../../easy_tpp/preprocess/event_tokenizer.py).



An example
----------------

We take a real event sequence for example. Assume we have an input sequence $[ 1,  9,  5,  0]$ with num_event_types=11 and max_len=6. 

Then the padded time_seqs, time_delta_seqs and type_seqs become

.. code-block:: bash

    # time_seqs
    [ 0.0000,  0.8252,  1.3806,  1.8349, 11.0000, 11.0000]

    # time_delta_seqs
    [ 0.0000,  0.8252,  0.5554,  0.4542, 11.0000, 11.0000]

    # type_seqs
    [ 1,  9,  5,  0, 11, 11]


The mask sequences are 

.. code-block:: bash

    # batch_non_pad_mask
    [ True,  True,  True,  True, False, False]

    # attention_mask
    [[True,  True,  True,  True,  True,  True],
    [False,  True,  True,  True,  True,  True],
    [False, False,  True,  True,  True,  True],
    [False, False, False,  True,  True,  True],
    [False, False, False, False,  True,  True],
    [False, False, False, False,  True,  True]]

    # type_mask
    [[False,  True, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, True, False],
    [False, False, False, False, False,  True, False, False, False, False, False],
    [True, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False]],


The runnable examples of constructing and iterating the dataset object can be found at `event_tokenizer <../../examples/dataset/event_tokenizer.py>`_


 




