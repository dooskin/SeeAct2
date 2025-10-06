from typing import Optional


class RetryableException(Exception):
    """Base exception class for errors that can be retried."""
    pass

class CriticalAgentError(Exception):
    """Exception for critical agent errors that should not be retried."""
    pass
    
class TaskExecutionRetryError(RetryableException):
    """Custom exception for task execution errors."""
    def __init__(self, task_id: str, message: str, context: Optional[str] = ""):
        super().__init__(f"Error in task {task_id}: {message}. From: {context}")
        self.task_id = task_id
        self.message = message
        self.context = context
    def __str__(self) -> str:
        return f"TaskExecutionError in {self.task_id}: {self.message}, From: {self.context}"