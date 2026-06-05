import os
import tempfile

from evaluation.experiment_tracking import ExperimentTracker


class TestExperimentTracker:
    def test_save_and_load_experiment(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tracker = ExperimentTracker(db_path=db_path)
            config = {"victim": "KeywordFilter", "keywords": ["bomb"]}
            tracker.save_experiment("exp_001", config)
            loaded = tracker.load_experiment("exp_001")
            assert loaded is not None
            assert loaded["id"] == "exp_001"
            assert loaded["config"] == config
            assert loaded["status"] == "running"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_save_with_results(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tracker = ExperimentTracker(db_path=db_path)
            config = {"victim": "RegexVictim", "pattern": r"\d+"}
            results = {"accuracy": 0.95, "f1": 0.92}
            tracker.save_experiment("exp_002", config)
            tracker.save_experiment("exp_002", config, results)
            loaded = tracker.load_experiment("exp_002")
            assert loaded["results"] == results
            assert loaded["status"] == "completed"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_list_experiments(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tracker = ExperimentTracker(db_path=db_path)
            tracker.save_experiment("exp_a", {"key": "a"})
            tracker.save_experiment("exp_b", {"key": "b"})
            experiments = tracker.list_experiments()
            ids = [e["id"] for e in experiments]
            assert "exp_a" in ids
            assert "exp_b" in ids
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_load_nonexistent_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tracker = ExperimentTracker(db_path=db_path)
            assert tracker.load_experiment("nonexistent") is None
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_delete_experiment(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tracker = ExperimentTracker(db_path=db_path)
            tracker.save_experiment("exp_del", {"key": "value"})
            assert tracker.load_experiment("exp_del") is not None
            tracker.delete_experiment("exp_del")
            assert tracker.load_experiment("exp_del") is None
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_git_hash_is_recorded(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            tracker = ExperimentTracker(db_path=db_path)
            tracker.save_experiment("exp_git", {"test": True})
            loaded = tracker.load_experiment("exp_git")
            assert loaded["git_hash"] is not None
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
