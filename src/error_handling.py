"""
Enhanced error handling utilities for Mistral Docs Assistant.
"""
import logging
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AppError:
    """Structured error object for tracking application errors."""
    error_type: str
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "stack_trace": self.stack_trace
        }
    
    @classmethod
    def from_exception(cls, error_type: str, exception: Exception, details: Optional[Dict[str, Any]] = None) -> 'AppError':
        """Create an AppError from an exception."""
        return cls(
            error_type=error_type,
            message=str(exception),
            timestamp=datetime.now(),
            details=details,
            stack_trace=traceback.format_exc()
        )

class ErrorHandler:
    """Handle and track application errors."""
    
    def __init__(self):
        self.errors: List[AppError] = []
        self.max_errors = 100  # Maximum number of errors to keep in memory
    
    def log_error(self, error: AppError) -> None:
        """Log an error and add it to the error list."""
        logger.error(f"{error.error_type}: {error.message}")
        if error.details:
            logger.debug(f"Error details: {error.details}")
        
        # Add to error list, maintaining max size
        self.errors.append(error)
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)  # Remove oldest error
    
    def capture_exception(self, error_type: str, exception: Exception, details: Optional[Dict[str, Any]] = None) -> AppError:
        """Capture an exception, log it, and return the error object."""
        error = AppError.from_exception(error_type, exception, details)
        self.log_error(error)
        return error
    
    def get_recent_errors(self, count: int = 5) -> List[AppError]:
        """Get the most recent errors."""
        return self.errors[-count:] if self.errors else []
    
    def get_errors_by_type(self, error_type: str) -> List[AppError]:
        """Get errors filtered by type."""
        return [error for error in self.errors if error.error_type == error_type]
    
    def clear_errors(self) -> None:
        """Clear all errors."""
        self.errors = []

# Create a global error handler instance
error_handler = ErrorHandler()

def safe_execute(func, error_type: str, *args, **kwargs):
    """
    Execute a function safely, capturing any exceptions.
    
    Args:
        func: The function to execute
        error_type: Type of error for categorization
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Tuple of (result, error). If successful, error is None.
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error = error_handler.capture_exception(error_type, e, {
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        })
        return None, error

def get_user_friendly_error_message(error_type: str) -> str:
    """Get a user-friendly error message based on error type."""
    error_messages = {
        "document_processing": "There was an issue processing your document. Please check the file format and try again.",
        "model_loading": "The AI model couldn't be loaded. We've switched to a backup model to continue.",
        "vector_store": "There was a problem indexing your documents. Try processing them again with different settings.",
        "api": "We couldn't connect to the AI service. Please check your internet connection and try again.",
        "query_processing": "I had trouble understanding your question. Could you try rephrasing it?",
        "memory": "I encountered a memory issue. Try clearing the conversation and starting again."
    }
    
    return error_messages.get(error_type, "An unexpected error occurred. Please try again.")
