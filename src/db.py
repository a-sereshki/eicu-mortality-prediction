"""
Database connection utilities for the eICU mortality prediction project.

Provides a single function `get_engine()` that returns a SQLAlchemy engine
connected to the local PostgreSQL eICU database. Reused across notebooks
and scripts to avoid hardcoding connection strings.
"""

from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Local PostgreSQL connection parameters
# NOTE: password is hardcoded for local development only
# Will be moved to environment variable before repo goes public for recruiters
DB_USER = "postgres"
DB_PASSWORD = "***REMOVED***"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "eicu"


def get_engine() -> Engine:
    """Return a SQLAlchemy engine connected to the local eICU database."""
    # URL-encode the password so special characters (@, :, /, etc.) don't
    # break the connection URL parsing
    encoded_password = quote_plus(DB_PASSWORD)

    connection_string = (
        f"postgresql+psycopg2://{DB_USER}:{encoded_password}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


if __name__ == "__main__":
    # Quick connection test when this file is run directly
    import pandas as pd

    engine = get_engine()
    result = pd.read_sql("SELECT COUNT(*) AS n FROM cohort;", engine)
    print(f"Connection successful. Cohort size: {result.iloc[0]['n']} patients.")
