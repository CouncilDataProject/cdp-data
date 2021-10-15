from cdp_backend.database import models as db_models
from cdp_backend.pipeline.transcript_model import Transcript
import fireo
import re
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client

# Connect to the database
fireo.connection(client=Client(
    project="cdp-seattle-staging-dbengvtn",
    credentials=AnonymousCredentials()
))

models_to_get = [
    db_models.Body,
    db_models.Event,
    db_models.EventMinutesItem,
    # This is an enum
    # db_models.EventMinutesItemDecision,
    db_models.EventMinutesItemFile,
    db_models.File,
    # Do not include indexed models that are used for search
    db_models.Matter,
    db_models.MatterFile,
    db_models.MatterSponsor,
    db_models.MatterStatus,
    # This is an enum
    # db_models.MatterStatusDecision,
    db_models.MinutesItem,
    db_models.Person,
    db_models.Role,
    db_models.Seat,
    db_models.Session,
    db_models.Transcript,
    db_models.Vote,
    # This is an enum
    # db_models.VoteDecision
]

def process_model(model):
    all_items = model.collection.fetch()
    items_list = list(map(process_model_item, all_items))

    return (model.collection_name, pd.DataFrame(items_list))

def process_model_item(item):
    return dict(map(process_model_field, item.to_dict().items()))

def process_model_field(key_value_pair):
    key = key_value_pair[0]
    value = key_value_pair[1]

    if isinstance(value, fireo.queries.query_wrapper.ReferenceDocLoader):
        key = re.sub('(_ref)?$', '_id', key, 1)
        value = value.ref.id

    return (key, value)


results = dict(map(process_model, models_to_get))

# CSV output
base_dir = Path(__file__).parent
(base_dir / 'csv').mkdir(exist_ok=True)
for key, value in results.items():
    with open(base_dir / 'csv' / (key + '.csv'), 'w') as f:
        f.write(value.to_csv(index=False))

# SQLite output
engine = create_engine('sqlite:///' + str(base_dir / 'cdp_data.db'), echo=False)

for key, value in results.items():
    if len(value.columns) > 0:
        value.to_sql(key, con=engine, index=False, if_exists='replace')
