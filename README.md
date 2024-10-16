# Black Forest Labs API Client

This is an unofficial Python client for interacting with the
[Black Forest Labs API](https://api.bfl.ml/).

```console
$ export BFL_API_KEY="..."
```

```python
import black_forest_labs
import time

task = black_forest_labs.generate(
    "flux-pro",
    prompt="ein fantastisches bild",
)

print(f"Task created with id `{task.id}`")

print("Polling for results...", end="", flush=True)
while not task.is_done:
    print(".", end="", flush=True)
    task = black_forest_labs.get_result(task.id)
    time.sleep(1)

print(f"\nTask finished with status `{task.status}`")
if task.result:
    print(f"Prompt: {task.result.prompt}")
    print(f"Image URL: {task.result.sample}")

```

<output>

```
Task created with id `ce9e065d-dc90-4633-9c1e-44ea839ed569`
Polling for results............
Task finished with status `Ready`
Prompt: ein fantastisches bild
Image URL: https://bflapistorage.blob.core.windows.net/public/db43e36806f74c1a9c6972127c9d71ea/sample.jpg
```

</output>
