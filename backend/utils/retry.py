import asyncio
import logging
import random
from functools import wraps

import httpx
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


def retry(max_retry=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):

            last_error = None

            for attempt in range(max_retry):
                try:
                    logger.info(f"Attempt {attempt + 1}/{max_retry}")

                    async for chunk in func(*args, **kwargs):
                        yield chunk

                    return

                except (TimeoutError, httpx.TimeoutException) as e:
                    logger.warning("Request timed out!")
                    last_error = e

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code

                    if status_code == 429:
                        logger.warning("Rate limit exceeded!")
                        last_error = e

                    elif 500 <= status_code <= 599:
                        logger.warning(f"Server error: {status_code}")
                        last_error = e

                    else:
                        raise

                # Final attempt failed
                if attempt == max_retry - 1:
                    logger.error("All attempts exhausted!")

                    if isinstance(
                        last_error,
                        (TimeoutError, httpx.TimeoutException),
                    ):
                        raise HTTPException(
                            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail="LLM request timed out after all retry attempts.",
                        ) from last_error

                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="LLM provider is unavailable after all retry attempts.",
                    ) from last_error

                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt)
                delay += random.uniform(0, 0.5)

                logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        return wrapper
    return decorator
