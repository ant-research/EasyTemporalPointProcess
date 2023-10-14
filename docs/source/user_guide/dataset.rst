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

Finally, each batch contains six elements: time_seqs, time_delta_seqs, event_seq, batch_non_pad_mask, attention_mask, type_mask. The implementation of padding mechanism can be found at `event_tokenizer <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/easy_tpp/preprocess/event_tokenizer.py>`_.



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


The runnable examples of constructing and iterating the dataset object can be found at `examples/event_tokenizer.py <https://github.com/ant-research/EasyTemporalPointProcess/blob/main/examples/event_tokenizer.py>`_


Preprocessed Datasets
===================================

We have preprocessed some widely-used open source datasets in Gatech format, which can be found at `Google Drive <https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w>`_. We use them for validating and benchmarking EasyTPP models.

- Retweet (`Zhou, 2013 <http://proceedings.mlr.press/v28/zhou13.pdf>`_). This dataset contains time-stamped user retweet event sequences.  The events are categorized into 3 types: retweets by “small,” “medium” and “large” users. Small users have fewer than 120 followers, medium users have fewer than 1363, and the rest are large users. We work on a subset of 5200 most active users with an average sequence length of 70.
- Taxi (`Whong, 2014 <https://chriswhong.com/open-data/foil_nyc_taxi>`_). This dataset tracks the time-stamped taxi pick-up and drop-off events across the five boroughs of the New York City; each (borough, pick-up or drop-off) combination defines an event type, so there are 10 event types in total. We work on a randomly sampled subset of 2000 drivers and each driver has a sequence. We randomly sampled disjoint train, dev and test sets with 1400, 200 and 400 sequences.
- StackOverflow ( `Leskovec, 2014 <https://snap.stanford.edu/data/>`_). This dataset has two years of user awards on a question-answering website: each user received a sequence of badges and there are 22 different kinds of badges in total. We randomly sampled disjoint train, dev and test sets with 1400,400 and 400 sequences from the dataset.
- Taobao (`Xue et al, 2022 <https://arxiv.org/abs/2210.01753>`_). This dataset contains time-stamped user click behaviors on Taobao shopping pages from November 25 to December 03, 2017. Each user has a sequence of item click events with each event containing the timestamp and the category of the item. The categories of all items are first ranked by frequencies and the top 19 are kept while the rest are merged into one category, with each category corresponding to an event type. We work on a subset of 4800 most active users with an average sequence length of 150 and then end up with 20 event types.
- Amazon (`Xue et al, 2022 <https://arxiv.org/abs/2210.01753>`_). This dataset includes time-stamped user product reviews behavior from January, 2008 to October, 2018. Each user has a sequence of produce review events with each event containing the timestamp and category of the reviewed product, with each category corresponding to an event type. We work on a subset of 5200 most active users with an average sequence length of 70 and then end up with 16 event types.

Besides, we also published two textual event sequence datasets:

- GDELT (`Shi et al, 2023  <https://arxiv.org/abs/2305.16646>`_). The GDELT Project monitors events all over the world, with live datasets updated every 15 minutes. We only focused on the political events that happened in G20 countries from 2022-01-01 to 2022-07-31, ending up with a corpus of 109000 time-stamped event tokens. The event type of each token has a structured name of the format subject-predicate-object. Each {predicate} is one of the twenty CAMEO codes such as {CONSULT} and {INVESTIGATE}; each {subject} or {object} is one of the 2279 political entities (individuals, groups, and states) such as {Tesla} and {Australia}. We split the dataset into disjoint train, dev, and test sets based on their dates: the 83100 events that happened before 2022-07-05 are training data; the 16650 events after 2022-07-19 are test data; the 9250 events between these dates are development data.
- Amazon-text-review (`Shi et al, 2023  <https://arxiv.org/abs/2305.16646>`_). This dataset contains user reviews on Amazon shopping website from 2014-01-04 to 2016-10-02. We focused on the most active 2500 users and each user has a sequence of product review events. The type is the category of the product: we selected the most frequently-reviewed 23 categories and grouped all the others into a special OTHER category, ending up with 24 categories in total. Each review event also has a mark which is the actual content of the review. Each of the 2500 sequences is cut into three segments: the events that happened before 2015-08-01 are training data; those after 2016-02-01 are test data; the events between these dates are dev data. Then we have 49,680 training tokens, 7,020 dev tokens, and 13,090 test tokens.
