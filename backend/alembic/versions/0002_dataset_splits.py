"""per-run dataset splits: split_source and given_splits on dataset_version

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("dataset_version") as batch:
        batch.add_column(sa.Column("split_source", sa.String(), nullable=False, server_default="none"))
        batch.add_column(sa.Column("given_splits", sa.JSON(), nullable=False, server_default="{}"))


def downgrade() -> None:
    with op.batch_alter_table("dataset_version") as batch:
        batch.drop_column("given_splits")
        batch.drop_column("split_source")
