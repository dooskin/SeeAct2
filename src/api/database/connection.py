"""
Unified database connection service for SeeAct2 API
"""
import os
from typing import Optional


class DatabaseConnection:
    """Unified database connection service"""
    
    def __init__(self):
        self._connection_string: Optional[str] = None
    
    @property
    def connection_string(self) -> str:
        """Get the database connection string"""
        if self._connection_string is None:
            # Try NEON_DATABASE_URL first (for compatibility with existing code)
            # then fall back to DATABASE_URL
            self._connection_string = (
                os.getenv("NEON_DATABASE_URL") or 
                os.getenv("DATABASE_URL") or 
                "postgresql://user:password@localhost:5432/seeact2"
            )
        return self._connection_string
    
    def get_connection_string(self) -> str:
        """Get the database connection string"""
        return self.connection_string


# Global database connection instance
db_connection = DatabaseConnection()


def get_database_connection() -> DatabaseConnection:
    """Get the global database connection instance"""
    return db_connection
