from .client import Client, Result, Status, Task

default_client = Client()

generate = default_client.generate
get_result = default_client.get_result

__all__ = ["Client", "Task", "Status", "Result", "generate", "get_result"]
