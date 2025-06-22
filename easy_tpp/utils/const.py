from enum import Enum


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    def __str__(self):
        return str(self.value)

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`EventTokenizer.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`EventTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    NUMPY = "np"


class RunnerPhase(ExplicitEnum):
    """Model runner phase enum.
    """
    TRAIN = 'train'
    VALIDATE = 'validate'
    PREDICT = 'predict'


class LossFunction(ExplicitEnum):
    """Loss function for neural TPP model.
    """
    LOGLIKE = 'loglike'
    PARTIAL_TIME_LOSS = 'rmse'
    PARTIAL_EVENT_LOSS = 'accuracy'


class LogConst:
    """Format for log handler.
    """
    DEFAULT_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
    DEFAULT_FORMAT_LONG = '%(asctime)s - %(filename)s[pid:%(process)d;line:%(lineno)d:%(funcName)s]' \
                          ' - %(levelname)s: %(message)s'


class PredOutputIndex:
    """Positional index for the output tuple in ModelRunner.
    """
    TimePredIndex = 0
    TypePredIndex = 1


class DefaultRunnerConfig:
    DEFAULT_DATASET_ID = 'conttime'


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`EventTokenizer.__call__`]. Useful for tab-completion in
    an IDE.
    """

    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class Backend(ExplicitEnum):
    """
    Possible values for the `backend` argument in configuration.
    """

    Torch = 'torch'
