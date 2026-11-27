"""merge parallel migration branches

Revision ID: 5d2bb2617176
Revises: 2dd40cc2eec7, b21a6e7026e0
Create Date: 2026-11-25 18:19:29.417176

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '5d2bb2617176'
down_revision: Union[str, Sequence[str], None] = ('2dd40cc2eec7', 'b21a6e7026e0')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
