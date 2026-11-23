"""initial

Revision ID: 0001
Revises: 
"""
import datetime as dt

import sqlalchemy as sa
from alembic import op


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('dataset',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('task', sa.String(), nullable=False),
    sa.Column('eval_only', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('gpu_lock',
    sa.Column('id', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('holder_kind', sa.String(), nullable=True),
    sa.Column('holder_id', sa.String(), nullable=True),
    sa.Column('acquired_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    gpu_lock = sa.table(
        "gpu_lock",
        sa.column("id", sa.Integer),
        sa.column("holder_kind", sa.String),
        sa.column("holder_id", sa.String),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    op.bulk_insert(
        gpu_lock,
        [{"id": 1, "holder_kind": None, "holder_id": None, "created_at": dt.datetime.now(dt.timezone.utc)}],
    )
    op.create_table('recipe',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('upload',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('filename', sa.String(), nullable=False),
    sa.Column('length', sa.BigInteger(), nullable=False),
    sa.Column('state', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('dataset_version',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('number', sa.Integer(), nullable=False),
    sa.Column('hash', sa.String(), nullable=False),
    sa.Column('path', sa.String(), nullable=False),
    sa.Column('n_samples', sa.Integer(), nullable=False),
    sa.Column('report', sa.JSON(), nullable=False),
    sa.Column('provenance', sa.JSON(), nullable=False),
    sa.Column('split_ratios', sa.JSON(), nullable=False),
    sa.Column('parent_version_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['dataset.id'], ),
    sa.ForeignKeyConstraint(['parent_version_id'], ['dataset_version.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id', 'number')
    )
    op.create_table('recipe_version',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('recipe_id', sa.Integer(), nullable=False),
    sa.Column('number', sa.Integer(), nullable=False),
    sa.Column('spec', sa.JSON(), nullable=False),
    sa.Column('yaml', sa.Text(), nullable=False),
    sa.Column('frozen', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['recipe_id'], ['recipe.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('recipe_id', 'number')
    )
    op.create_table('ingest',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('upload_id', sa.String(), nullable=False),
    sa.Column('dataset_name', sa.String(), nullable=False),
    sa.Column('task', sa.String(), nullable=False),
    sa.Column('eval_only', sa.Boolean(), nullable=False),
    sa.Column('provenance', sa.JSON(), nullable=False),
    sa.Column('state', sa.String(), nullable=False),
    sa.Column('report', sa.JSON(), nullable=True),
    sa.Column('version_id', sa.Integer(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['upload_id'], ['upload.id'], ),
    sa.ForeignKeyConstraint(['version_id'], ['dataset_version.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('run',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('recipe_version_id', sa.Integer(), nullable=False),
    sa.Column('state', sa.String(), nullable=False),
    sa.Column('attempt', sa.Integer(), nullable=False),
    sa.Column('resume', sa.Boolean(), nullable=False),
    sa.Column('cancel_requested', sa.Boolean(), nullable=False),
    sa.Column('run_dir', sa.String(), nullable=False),
    sa.Column('commercial_ok', sa.Boolean(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('oom', sa.Boolean(), nullable=False),
    sa.Column('git_sha', sa.String(), nullable=True),
    sa.Column('lock_hash', sa.String(), nullable=True),
    sa.Column('dataset_hashes', sa.JSON(), nullable=False),
    sa.Column('mlflow_run_id', sa.String(), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['recipe_version_id'], ['recipe_version.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('model_version',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('number', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=False),
    sa.Column('task', sa.String(), nullable=False),
    sa.Column('base_model', sa.String(), nullable=False),
    sa.Column('metrics', sa.JSON(), nullable=False),
    sa.Column('exports', sa.JSON(), nullable=False),
    sa.Column('commercial_ok', sa.Boolean(), nullable=False),
    sa.Column('champion', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['run.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'number'),
    sa.UniqueConstraint('run_id')
    )


def downgrade() -> None:
    op.drop_table('model_version')
    op.drop_table('run')
    op.drop_table('ingest')
    op.drop_table('recipe_version')
    op.drop_table('dataset_version')
    op.drop_table('upload')
    op.drop_table('recipe')
    op.drop_table('gpu_lock')
    op.drop_table('dataset')
