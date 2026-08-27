from datetime import datetime


DB_PATH = "youtube_trends.db"
DATA_START = '2026-07-26 14'  # Start of data to consider for backtesting
TRAIN_END = '2026-08-20 14' # End of training data for backtesting
TEST_END = '2026-08-27 08' # End of test data for backtesting
HORIZON_HOURS = 6  # Number of hours to forecast ahead for backtesting
BUCKET_HOURS = 6  # Number of hours per bucket (must divide HORIZON_HOURS evenly)

_origin = datetime.strptime(DATA_START, '%Y-%m-%d %H')
_train_end = datetime.strptime(TRAIN_END, '%Y-%m-%d %H')
_test_end = datetime.strptime(TEST_END, '%Y-%m-%d %H')

assert _train_end > _origin, "TRAIN_END must be after DATA_START"
assert _test_end > _train_end, "TEST_END must be after TRAIN_END"

for _name, _dt in (("TRAIN_END", _train_end), ("TEST_END", _test_end)):
    _hours = (_dt - _origin).total_seconds() / 3600
    assert _hours % BUCKET_HOURS == 0, \
        f"{_name}={_dt} is not on the bucket grid (origin={_origin}) BUCKET_HOURS={BUCKET_HOURS}."

assert HORIZON_HOURS % BUCKET_HOURS == 0, \
    f"HORIZON_HOURS={HORIZON_HOURS} not representable with BUCKET_HOURS={BUCKET_HOURS}."
# Number of buckets to forecast ahead for backtesting
HORIZON_BUCKETS = HORIZON_HOURS // BUCKET_HOURS

STEP = HORIZON_HOURS  # Step size for backtesting, in hours


assert STEP % BUCKET_HOURS == 0, \
    f"STEP={STEP} not representable with BUCKET_HOURS={BUCKET_HOURS}."
STEP_BUCKETS = STEP // BUCKET_HOURS  # Step size for backtesting, in buckets
# Minimum distinct number of hours (in the training window) a keyword must appear in to be considered
MIN_HOURS_PRESENT = 10
# Minimum number of unique videos a keyword must appear in to be considered
MIN_VIDEO_COUNT = 4
TOP_N = 50  # Number of top keywords to consider based on frequency
# CSV file containing labels for keywords
LABELS_CSV = "labels/backtest_labels.csv"