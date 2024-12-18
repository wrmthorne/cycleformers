from dataclasses import dataclass, field
from typing import Optional

import pytest
from prefix_dataclass import prefixed_view


# Test fixtures
@dataclass
class SimpleConfig:
    name: str | None = None
    value: int = 42


@dataclass
class ComplexConfig:
    name: str
    values: list[int] = field(default_factory=list)
    optional: Optional[str] = None


@dataclass(frozen=True)
class FrozenConfig:
    name: str
    value: int = 42


class TestPrefixedView:
    def test_basic_creation(self):
        """Test basic creation of prefixed view class"""

        @prefixed_view(SimpleConfig, "test_")
        class TestConfig:
            pass

        assert TestConfig.__annotations__ == {"test_name": str | None, "test_value": int}

    @pytest.mark.parametrize(
        "input_values,expected",
        [
            ({"test_name": "example"}, SimpleConfig(name="example", value=42)),
            ({"test_name": "test", "test_value": 100}, SimpleConfig(name="test", value=100)),
            ({}, SimpleConfig(name=None, value=42)),
        ],
    )
    def test_valid_instances(self, input_values, expected):
        """Test creation of instances with valid inputs"""

        @prefixed_view(SimpleConfig, "test_")
        class TestConfig:
            pass

        instance = TestConfig(**input_values)
        assert instance == expected

    @pytest.mark.parametrize(
        "prefix",
        [
            "test_",
            "_",
            "PREFIX_",
            "A_",
        ],
    )
    def test_prefix_styles(self, prefix):
        """Test different prefix styles"""

        @prefixed_view(SimpleConfig, prefix)
        class TestConfig:
            pass

        # Verify prefixed attributes exist
        assert all(name.startswith(prefix) for name in TestConfig.__annotations__)

    def test_default_factory(self):
        """Test handling of default_factory fields"""

        @prefixed_view(ComplexConfig, "test_")
        class TestConfig:
            pass

        instance = TestConfig(test_name="test")
        assert instance.values == []

        instance = TestConfig(test_name="test", test_values=[1, 2, 3])
        assert instance.values == [1, 2, 3]

    def test_frozen_dataclass(self):
        """Test with frozen dataclass"""

        @prefixed_view(FrozenConfig, "test_")
        class TestConfig:
            pass

        instance = TestConfig(test_name="test")
        assert instance.name == "test"
        assert instance.value == 42

    @pytest.mark.parametrize(
        "invalid_base,prefix,expected_error",
        [
            (None, "test_", TypeError),  # Missing base class
            (dict, "test_", TypeError),  # not a dataclass
        ],
    )
    def test_creation_errors(self, invalid_base, prefix, expected_error):
        """Test error cases for class creation"""
        with pytest.raises(expected_error):

            @prefixed_view(invalid_base, prefix)
            class TestConfig:
                pass

    def test_nested_dataclass(self):
        """Test handling of nested dataclasses"""

        @dataclass
        class NestedConfig:
            config: SimpleConfig
            name: str

        @prefixed_view(NestedConfig, "test_")
        class TestNestedConfig:
            pass

        nested = SimpleConfig(name="nested")
        instance = TestNestedConfig(test_config=nested, test_name="test")
        assert instance.config == nested
        assert instance.name == "test"

    def test_inheritance(self):
        """Test with inherited dataclass"""

        @dataclass
        class BaseConfig:
            name: str

        @dataclass
        class ChildConfig(BaseConfig):
            value: int = 42

        @prefixed_view(ChildConfig, "test_")
        class TestConfig:
            pass

        # Should include both inherited and child fields
        assert set(TestConfig.__annotations__.keys()) == {"test_name", "test_value"}
