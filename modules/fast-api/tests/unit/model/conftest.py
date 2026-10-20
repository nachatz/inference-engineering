import importlib.util

# model/ is build-time code: without PyTorch (as in the serving image) these tests are skipped.
collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
