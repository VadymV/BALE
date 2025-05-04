import logging
from collections import defaultdict
from pathlib import Path

from NEMO.nemo.epochs import get_epochs_dfs, get_epochs_raw_dataset
from NEMO.nemo.utils import save_to

from bale.dataset.helpers import CHANNELS, get_epochs
from bale.helpers.misc import set_logging
from bale import EVENTS_TYPE, PROJECT_PATH, TASK_TYPE

if __name__ == "__main__":
    set_logging(PROJECT_PATH.__str__())
    epochs_path = Path("{project_path}/data/nemo-processed-data".format(
        project_path=PROJECT_PATH.__str__()))

    epochs_df, epochs_metadata = get_epochs_dfs(get_epochs(epochs_path))
    Xr, y, epoch_ids = get_epochs_raw_dataset(
        epochs_df,
        include_events=EVENTS_TYPE,
        task=TASK_TYPE,
        channels=CHANNELS,
    )

    X = defaultdict(list)
    for subject, d in Xr.items():
        sXf = []
        X[subject] = d.transpose(0, 2, 1)  # epochs, channels, time

    features_path = Path("{project_path}/data/nemo".format(
        project_path=PROJECT_PATH.__str__()))
    features_path.mkdir(exist_ok=True)

    save_to(X, features_path / "X.pkl")
    save_to(y, features_path / "y.pkl")
    save_to(epoch_ids, features_path / "epoch_ids.pkl")
    save_to(CHANNELS, features_path / "channels.pkl")
    save_to(epochs_metadata, features_path / "metadata.pkl")

    logging.info("Features are saved into{}".format(features_path))
