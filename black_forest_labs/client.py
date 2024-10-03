import os
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict, Union, overload

import httpx
from pydantic import AfterValidator, BaseModel, HttpUrl
from typing_extensions import NotRequired, Unpack


class Client:
    """A client for the Black Forest Labs API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.bfl.ml",
    ) -> None:
        """Initialize the Black Forest Labs API client.

        Args:
            api_key: The API key for authentication. If not provided,
                it will be read from the BFL_API_KEY environment variable.
            base_url: The base URL for the API. Defaults to "https://api.bfl.ml".

        Raises:
            ValueError: If the API key is not provided and not set in the environment.
        """

        self._api_key = api_key or os.environ.get("BFL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key is required. Set BFL_API_KEY environment variable or pass it to the constructor."
            )

        self._client = httpx.Client(base_url=base_url)
        self._client.headers.update(
            {"x-key": self._api_key, "User-Agent": "bfl-python-client/0.1.0"}
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Make an HTTP request to the Black Forest Labs API.

        Args:
            method: The HTTP method to use (e.g., "GET", "POST").
            path: The API endpoint path.
            **kwargs: Additional arguments to pass to the request.

        Returns:
            The response from the API.

        Raises:
            httpx.HTTPStatusError: If the response contains an HTTP error status code.
        """

        resp = self._client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp

    def get_result(self, task: Union["Task", str]) -> "Task":
        """Get the result of a generation task.

        Args:
            task_id: The ID of the task to retrieve the result for.

        Returns:
            A Pydantic model containing the task result.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """

        task_id = task.id if isinstance(task, Task) else task
        resp = self._request("GET", "/v1/get_result", params={"id": task_id})
        return Task(**resp.json())

    @overload
    def generate(
        self, model: Literal["flux-pro-1.1"], **params: Unpack["FluxProPlusInputs"]
    ) -> "Task": ...

    @overload
    def generate(
        self, model: Literal["flux-pro"], **params: Unpack["FluxProInputs"]
    ) -> "Task": ...

    @overload
    def generate(
        self, model: Literal["flux-dev"], **params: Unpack["FluxDevInputs"]
    ) -> "Task": ...

    def generate(
        self, model: Literal["flux-pro-1.1", "flux-pro", "flux-dev"], **params: Any
    ) -> "Task":
        """Generate an image with the specified model.

        Args:
            model: The model to use for generation.
            **params: Model-specific parameters for image generation.

        Returns:
            A dictionary containing the task ID for the generated image.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """

        resp = self._request("POST", f"/v1/{model}", json=params)
        return Task(**resp.json())


class Status(Enum):
    """The status of an image generation task."""

    NOT_FOUND = "Task not found"
    PENDING = "Pending"
    REQUEST_MODERATED = "Request Moderated"
    CONTENT_MODERATED = "Content Moderated"
    READY = "Ready"
    ERROR = "Error"

    def __str__(self) -> str:
        return self.value


class Task(BaseModel):
    """An image generation task."""

    id: str
    status: Optional[Status] = None
    result: Optional["Result"] = None

    @property
    def is_done(self) -> bool:
        """Check if the task is finished."""
        return self.status in [
            Status.READY,
            Status.ERROR,
            Status.REQUEST_MODERATED,
            Status.CONTENT_MODERATED,
        ]


class Result(BaseModel):
    """The result of an image generation task."""

    sample: Annotated[HttpUrl, AfterValidator(str)]
    """The URL of the generated image."""

    prompt: str
    """The prompt used for image generation."""


class FluxProPlusInputs(TypedDict):
    """Input parameters for the FLUX 1.1 [pro] model.

    Attributes:
        prompt: Text prompt for image generation.
        width: Width of the generated image in pixels. Must be a multiple of 32. Default is 1024.
        height: Height of the generated image in pixels. Must be a multiple of 32. Default is 768.
        prompt_upsampling: Whether to perform upsampling on the prompt. Default is False.
        seed: Optional seed for reproducibility.
        safety_tolerance: Tolerance level for input and output moderation.
            Between 0 and 6, 0 being most strict, 6 being least strict. Default is 2.
    """

    prompt: str
    width: NotRequired[int]
    height: NotRequired[int]
    prompt_upsampling: NotRequired[bool]
    seed: NotRequired[int]
    safety_tolerance: NotRequired[int]


class FluxProInputs(FluxProPlusInputs):
    """Input parameters for the FLUX.1 [pro] model.

    Inherits all attributes from FluxProPlusInputs and adds:
        steps: Number of steps for the image generation process. Default is 40.
        guidance: Guidance scale for image generation. Default is 2.5.
        interval: Interval parameter for guidance control. Default is 2.0.
    """

    steps: NotRequired[int]
    guidance: NotRequired[float]
    interval: NotRequired[float]


class FluxDevInputs(FluxProInputs):
    """Input parameters for the FLUX.1 [dev] model.

    Inherits all attributes from FluxProInputs with some different defaults:
        steps: Number of steps for the image generation process. Default is 28.
        guidance: Guidance scale for image generation. Default is 3.0.
    """
