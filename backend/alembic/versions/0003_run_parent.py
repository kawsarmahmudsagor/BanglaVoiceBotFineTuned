"""warm start lineage: parent_run_id and parent_checkpoint on run

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("run") as batch:
        batch.add_column(sa.Column("parent_run_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("parent_checkpoint", sa.String(), nullable=True))
        batch.create_foreign_key("fk_run_parent_run_id", "run", ["parent_run_id"], ["id"])


def downgrade() -> None:
    with op.batch_alter_table("run") as batch:
        batch.drop_constraint("fk_run_parent_run_id", type_="foreignkey")
        batch.drop_column("parent_checkpoint")
        batch.drop_column("parent_run_id")
