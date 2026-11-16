
"""added chunk table

Revision ID: 975d7559b56d
Revises: 1ef499e99f6c
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
import sqlmodel

revision: str = '975d7559b56d'
down_revision: Union[str, Sequence[str], None] = '1ef499e99f6c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create Chunk table
    op.create_table(
        'chunk',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=False),
        sa.Column('text', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('filename', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column(
            'embedding',
            pgvector.sqlalchemy.vector.VECTOR(dim=768),
            nullable=False
        ),
        sa.ForeignKeyConstraint(
            ['document_id'],
            ['document.id'],
            ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_index(
        op.f('ix_chunk_document_id'),
        'chunk',
        ['document_id'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_chunk_document_id'),
        table_name='chunk'
    )
    op.drop_table('chunk')

    # Keep vector extension installed.