import json
import os
import tempfile

import pytest

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import IfThenElseNode, Program

from adapters.toy_victims.benchmark_generator import (
    ProgramDrivenVictim,
    generate_benchmark,
    generate_random_program,
    victim_from_program,
)


class TestGenerateRandomProgram:
    def test_generates_valid_program(self):
        program = generate_random_program(seed=42)
        assert isinstance(program, Program)
        assert isinstance(program.root, IfThenElseNode)
        assert program.complexity() > 0

    def test_different_seeds_different_programs(self):
        p1 = generate_random_program(seed=1)
        p2 = generate_random_program(seed=2)
        assert p1 != p2 or p1.complexity() != p2.complexity()

    def test_same_seed_same_program(self):
        p1 = generate_random_program(seed=42)
        p2 = generate_random_program(seed=42)
        assert p1 == p2

    def test_program_is_executable(self):
        program = generate_random_program(seed=42)
        executor = ProgramExecutor(default_registry)
        result = executor.execute(program, "test prompt")
        assert result in (0, 1)


class TestVictimFromProgram:
    def test_creates_executable_victim(self):
        program = generate_random_program(seed=42)
        victim = victim_from_program(program, name="test_victim")
        assert isinstance(victim, ProgramDrivenVictim)
        assert victim.respond("hello") in (0, 1)
        assert victim.get_ground_truth_program() is program


class TestProgramDrivenVictim:
    def test_respond_matches_program(self):
        program = generate_random_program(seed=42)
        victim = ProgramDrivenVictim(program=program)
        executor = ProgramExecutor(default_registry)
        assert victim.respond("test") == executor.execute(program, "test")

    def test_get_metadata(self):
        program = generate_random_program(seed=42)
        victim = ProgramDrivenVictim(program=program, name="custom")
        meta = victim.get_metadata()
        assert meta.get("has_ground_truth") is True


class TestGenerateBenchmark:
    def test_generates_benchmark_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifests = generate_benchmark(size=3, output_dir=tmpdir, seed=42)
            assert len(manifests) == 3
            for i in range(3):
                filepath = os.path.join(tmpdir, f"victim_{i}.json")
                assert os.path.exists(filepath)
                with open(filepath) as f:
                    data = json.load(f)
                assert "program_id" in data
            manifest_path = os.path.join(tmpdir, "manifest.json")
            assert os.path.exists(manifest_path)

    def test_reproducible_with_seed(self):
        with tempfile.TemporaryDirectory() as d1:
            with tempfile.TemporaryDirectory() as d2:
                m1 = generate_benchmark(size=2, output_dir=d1, seed=123)
                m2 = generate_benchmark(size=2, output_dir=d2, seed=123)
                assert m1[0]["program"] is not None
                assert m2[0]["program"] is not None
                assert m1[0]["complexity"] == m2[0]["complexity"]
