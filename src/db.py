"""
Database connection utilities for the eICU mortality prediction project.

Reads PostgreSQL connection parameters from a .env file at the project root.
The .env file must not be committed to version control.
"""

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Load .env from the project root, overriding any stale shell env vars
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


def get_engine() -> Engine:
    """Return a SQLAlchemy engine connected to the local eICU database.

    Requires DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME to be set
    in the .env file at the project root.
    """
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_PASSWORD"]
    db_host = os.environ["DB_HOST"]
    db_port = os.environ["DB_PORT"]
    db_name = os.environ["DB_NAME"]

    # URL-encode the password so special characters don't break URL parsing
    encoded_password = quote_plus(db_password)

    connection_string = (
        f"postgresql+psycopg2://{db_user}:{encoded_password}"
        f"@{db_host}:{db_port}/{db_name}"
    )
    return create_engine(connection_string)


if __name__ == "__main__":
    import pandas as pd

    engine = get_engine()
    result = pd.read_sql("SELECT COUNT(*) AS n FROM cohort;", engine)
    print(f"Connection successful. Cohort size: {result.iloc[0]['n']} patients.")
