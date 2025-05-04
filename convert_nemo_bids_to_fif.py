"""
Converts BIDS data to FIF data.
"""

import logging
import pandas as pd
import os
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids

import mne
from NEMO.nemo.epochs import create_epochs_from_raw
from NEMO.nemo.process_raw import process_raw
from NEMO.nemo.utils import get_all_subjects

EVENTS_TYPE = "empe"
PROJECT_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

def read_events_metadata(bids_path, subject, include_events):
    """
    Reads a subject's raw optical density data from BIDS and returns it as a dataframe.
    """
    subject_id = subject[-3:]

    sub_events_path = os.path.join(bids_path, "sub-{}".format(subject_id),
                                   'nirs', "sub-{}_task-{}_events.tsv".format(
            subject_id, include_events))

    return pd.read_csv(sub_events_path, sep="\t")


def read_raw_od(bids_path, subject, include_events):
    """
    Reads a subject's raw optical density data from BIDS and returns it as an MNE object.
    """
    subject_id = subject[-3:]
    bidspath = BIDSPath(
        subject=subject_id,
        task=include_events,
        root=bids_path,
        datatype="nirs",
    )
    # mne_bids does not currently support reading fnirs_od, so we have to manually set the channel types and ignore warnings
    with mne.utils.use_log_level("ERROR"):
        raw_od_bids = read_raw_bids(bidspath).load_data()
        ch_map = {ch: "fnirs_od" for ch in raw_od_bids.ch_names}
        raw_od_bids.set_channel_types(ch_map)
    return raw_od_bids

if __name__ == "__main__":

    bids_path = Path("{project_path}/data/nemo-bids".format(
        project_path=PROJECT_PATH.__str__()))
    epochs_path = Path("{project_path}/data/nemo-processed-data".format(
        project_path=PROJECT_PATH.__str__()))
    epochs_path.mkdir(exist_ok=True)

    for subject in get_all_subjects():
        logging.info(
            f"Processing subject={subject}, events={EVENTS_TYPE}")
        event_metadata = read_events_metadata(bids_path=bids_path,
                                              subject=subject,
                                              include_events=EVENTS_TYPE)
        raw_od = read_raw_od(bids_path, subject, include_events=EVENTS_TYPE)
        raw_haemo = process_raw(raw_od)
        events, event_name_mapping = mne.events_from_annotations(
            raw_haemo)
        epochs = create_epochs_from_raw(
            raw_haemo,
            events=events,
            event_metadata=event_metadata,
            event_name_mapping=event_name_mapping,
        )
        output_path = "{epochs_path}/{subject}_task-{inc_events}_epo.fif".format(
            epochs_path=epochs_path,
            subject=subject,
            inc_events=EVENTS_TYPE)
        epochs.save(
            output_path,
            overwrite=True,
            verbose="WARNING",
        )
        logging.info("Epochs are saved into {}".format(output_path))
