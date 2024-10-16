import httpx
import pytest
import respx

from black_forest_labs import Client, Result, Status, Task


@pytest.fixture
def mock_api_key():
    return "test_api_key"


@pytest.fixture
def client(mock_api_key):
    return Client(api_key=mock_api_key)


@respx.mock
def test_client_initialization(mock_api_key):
    client = Client(api_key=mock_api_key)
    assert client._api_key == mock_api_key
    assert isinstance(client._client, httpx.Client)
    assert client._client.headers["x-key"] == mock_api_key
    assert client._client.headers["User-Agent"] == "bfl-python-client/0.1.0"


def test_client_initialization_from_env(monkeypatch):
    monkeypatch.setenv("BFL_API_KEY", "env_api_key")
    client = Client()
    assert client._api_key == "env_api_key"


@respx.mock
def test_get_result(client):
    task_id = "test_task_id"
    mock_response = {
        "id": task_id,
        "status": "Ready",
        "result": {"sample": "https://example.com/image.png", "prompt": "Test prompt"},
    }

    respx.get(f"https://api.bfl.ml/v1/get_result?id={task_id}").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = client.get_result(task_id)
    assert isinstance(result, Task)
    assert result.id == task_id
    assert result.status == Status.READY
    assert isinstance(result.result, Result)
    assert result.result.sample == "https://example.com/image.png"
    assert result.result.prompt == "Test prompt"


@respx.mock
@pytest.mark.parametrize("model", ["flux-pro-1.1", "flux-pro", "flux-dev"])
def test_generate(client, model):
    task_id = "generated_task_id"
    mock_response = {"id": task_id}

    respx.post(f"https://api.bfl.ml/v1/{model}").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    params = {"prompt": "Test prompt"}
    result = client.generate(model, **params)

    assert isinstance(result, Task)
    assert result.id == task_id


@respx.mock
def test_request_error_handling(client):
    respx.get("https://api.bfl.ml/v1/get_result").mock(
        return_value=httpx.Response(404, json={"error": "Not found"})
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.get_result("non_existent_task")


def test_task_is_done():
    task = Task(id="test_task")

    task.status = Status.PENDING
    assert not task.is_done

    task.status = Status.READY
    assert task.is_done

    task.status = Status.ERROR
    assert task.is_done

    task.status = Status.REQUEST_MODERATED
    assert task.is_done

    task.status = Status.CONTENT_MODERATED
    assert task.is_done
