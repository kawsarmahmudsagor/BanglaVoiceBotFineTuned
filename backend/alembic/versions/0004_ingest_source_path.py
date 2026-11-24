"""local-path ingest: ingest.upload_id nullable, ingest.source_path

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("ingest") as batch:
        batch.alter_column("upload_id", existing_type=sa.String(), nullable=True)
        batch.add_column(sa.Column("source_path", sa.String(), nullable=True))


def downgrade() -> None:
    # Local-path ingests have no upload; they cannot exist under the old NOT NULL schema.
    op.execute("DELETE FROM ingest WHERE upload_id IS NULL")
    with op.batch_alter_table("ingest") as batch:
        batch.drop_column("source_path")
        batch.alter_column("upload_id", existing_type=sa.String(), nullable=False)
