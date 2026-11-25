
"""updated feedback model and request model to limit comment size

Revision ID: 2dd40cc2eec7
Revises: b628c23fb052
"""

from alembic import op

revision = "2dd40cc2eec7"
down_revision = "b628c23fb052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(
        "ix_llm_feedback_record_id",
        table_name="llm_feedback"
    )

    op.create_index(
        "ix_llm_feedback_record_id",
        "llm_feedback",
        ["record_id"],
        unique=True
    )


def downgrade() -> None:
    op.drop_index(
        "ix_llm_feedback_record_id",
        table_name="llm_feedback"
    )

    op.create_index(
        "ix_llm_feedback_record_id",
        "llm_feedback",
        ["record_id"],
        unique=False
    )
