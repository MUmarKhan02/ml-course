def accuracy(correct: int, total: int) -> float:
    """Return the share of correct predictions (0.0 to 1.0)."""
    if total == 0:
        return 0.0
    return correct / total

print(f"Accuracy: {accuracy(87, 100):.0%}")

def describe_model(name: str, version: int = 1) -> str:
    return f"{name} v{version}"

print(describe_model("churn-model"), "|", describe_model("churn-model", version=3))
print("Docstring:", accuracy.__doc__)

def average(*numbers: float) -> float:
    return sum(numbers) / len(numbers)

print(average(80, 90, 100))

def train(model_name: str, *datasets: str, epochs: int = 10, **hyperparams) -> str:
    return f"train {model_name} on {datasets} for {epochs} epochs with {hyperparams}"

print(train("bert", "imdb", "sst2", epochs=3, lr=2e-5, batch_size=32))

config = {"lr": 0.01, "dropout": 0.2}
print(train("mlp", "churn", **config))

def shout(text: str) -> str:
    return text.upper()

def apply_to_all(func, items):
    return [func(x) for x in items]

print(apply_to_all(shout, ["python", "sql"]))
print(apply_to_all(lambda x: x * 2, [1, 2, 3]))

def make_scaler(mean: float, std: float):
    def scale(x: float) -> float:
        return (x - mean) / std
    return scale

z = make_scaler(mean=50, std=10)
print("z(70) =", z(70), "| z(40) =", z(40))




import functools
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("day2")

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("%s took %.1f ms", func.__name__, elapsed_ms)
        return result
    return wrapper

@timer
def slow_sum(n: int) -> int:
    return sum(i * i for i in range(n))

print("slow_sum:", slow_sum(1_000_000))
print("Name kept by functools.wraps:", slow_sum.__name__)

import random

def retry(times: int = 3, base_delay: float = 0.01):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except ConnectionError as exc:
                    if attempt == times:
                        raise
                    log.warning("%s failed (%s); retry %d/%d", func.__name__, exc, attempt, times)
                    time.sleep(base_delay * 2 ** (attempt - 1) + random.uniform(0, base_delay))
        return wrapper
    return decorator

calls = {"n": 0}

@retry(times=4)
def call_model_api() -> str:
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("503 Service Unavailable")
    return "200 OK"

print("API result:", call_model_api())

def fib_slow(n: int) -> int:
    return n if n < 2 else fib_slow(n - 1) + fib_slow(n - 2)

@functools.lru_cache(maxsize=None)
def fib_fast(n: int) -> int:
    return n if n < 2 else fib_fast(n - 1) + fib_fast(n - 2)

t0 = time.perf_counter()
fib_slow(27)
t1 = time.perf_counter()
fib_fast(27)
t2 = time.perf_counter()
print(f"fib(27): slow {1000 * (t1 - t0):.0f} ms vs cached {1000 * (t2 - t1):.3f} ms")
print("fib_fast(80) =", fib_fast(80))

def count_up_to(n: int):
    i = 1
    while i <= n:
        yield i
        i += 1

gen = count_up_to(3)
print(next(gen), next(gen), next(gen))

def batch_iter(data: list, batch_size: int):
    for start in range(0, len(data), batch_size):
        yield data[start:start + batch_size]

for i, batch in enumerate(batch_iter(list(range(10)), batch_size=4)):
    print(f"batch {i}: {batch}")

def read_errors(lines):
    for line in lines:
        level, _, msg = line.partition(":")
        if level == "ERROR":
            yield msg.strip()

log_lines = ["INFO: started", "ERROR: GPU out of memory", "INFO: epoch 1", "ERROR: NaN loss"]
print("Errors:", list(read_errors(log_lines)))

squares_list = [x * x for x in range(100_000)]
squares_gen = (x * x for x in range(100_000))
print(f"list: {sys.getsizeof(squares_list):,} bytes | generator: {sys.getsizeof(squares_gen)} bytes")

with open("run_notes.txt", "w") as f:
    f.write("experiment: baseline-logreg\n")
with open("run_notes.txt") as f:
    print("File says:", f.read().strip())

from contextlib import contextmanager

@contextmanager
def experiment(name: str):
    run = {"name": name, "metrics": {}}
    start = time.perf_counter()
    log.info("START %s", name)
    try:
        yield run
    finally:
        log.info("END %s metrics=%s (%.2fs)", name, run["metrics"], time.perf_counter() - start)

with experiment("baseline-logreg") as run:
    run["metrics"]["auc"] = 0.87

# Exercise 1: count how many times a function is called
def count_calls(func):
    # @wraps copies func's name/docstring onto wrapper so it still looks like the original
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1               # bump the counter on every call
        return func(*args, **kwargs)     # run the real function and pass its result through
    wrapper.calls = 0                    # counter starts at zero, stored on the wrapper itself
    return wrapper                       # the decorated function is now this wrapper

@count_calls
def predict(x):
    return x > 0.5

predict(0.7)                             # calls = 1
predict(0.2)                             # calls = 2
assert predict.calls == 2

# Exercise 2: cut text into chunks of `size` characters
def chunked(text, size):
    # step through the string in jumps of `size`: 0, size, 2*size, ...
    for i in range(0, len(text), size):
        yield text[i:i + size]           # yield one slice at a time (the last may be shorter)

assert list(chunked("abcdefg", 3)) == ["abc", "def", "g"]

# Exercise 3: divide safely, returning a default instead of crashing
def safe_divide(a, b, default=0.0):
    try:
        return a / b                     # normal case
    except ZeroDivisionError:            # only catch divide-by-zero, so other bugs still show up
        return default                   # fall back to the default value

assert safe_divide(10, 2) == 5 and safe_divide(1, 0) == 0.0
print("All exercises passed!")           # only reached if every assert above passed