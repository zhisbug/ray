import logging

from typing import Dict, Any, List, Optional, Tuple

from ray._raylet import (
    Count as CythonCount,
    Histogram as CythonHistogram,
    Gauge as CythonGauge,
)  # noqa: E402

logger = logging.getLogger(__name__)


class Metric:
    """The parent class of custom metrics.

    Ray's custom metrics APIs are rooted from this class and share
    the same public methods.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 tag_keys: Optional[Tuple[str]] = None):
        if len(name) == 0:
            raise ValueError("Empty name is not allowed. "
                             "Please provide a metric name.")
        self._name = name
        self._description = description
        # The default tags key-value pair.
        self._default_tags = {}
        # Keys of tags.
        self._tag_keys = tag_keys or tuple()
        # The Cython metric class. This should be set in the child class.
        self._metric = None

        if not isinstance(self._tag_keys, tuple):
            raise TypeError("tag_keys should be a tuple type, got: "
                            f"{type(self._tag_keys)}")

        for key in self._tag_keys:
            if not isinstance(key, str):
                raise TypeError(f"Tag keys must be str, got {type(key)}.")

    def set_default_tags(self, default_tags: Dict[str, str]):
        """Set default tags of metrics.

        Example:
            >>> # Note that set_default_tags returns the instance itself.
            >>> counter = Counter("name")
            >>> counter2 = counter.set_default_tags({"a": "b"})
            >>> assert counter is counter2
            >>> # this means you can instantiate it in this way.
            >>> counter = Counter("name").set_default_tags({"a": "b"})

        Args:
            default_tags(dict): Default tags that are
                used for every record method.

        Returns:
            Metric: it returns the instance itself.
        """
        for key, val in default_tags.items():
            if key not in self._tag_keys:
                raise ValueError(f"Unrecognized tag key {key}.")
            if not isinstance(val, str):
                raise TypeError(f"Tag values must be str, got {type(val)}.")

        self._default_tags = default_tags
        return self

    def record(self,
               value: float,
               tags: Dict[str, str] = None,
               _internal=False) -> None:
        """Record the metric point of the metric.

        Tags passed in will take precedence over the metric's default tags.

        Args:
            value(float): The value to be recorded as a metric point.
        """
        assert self._metric is not None
        if isinstance(self._metric, CythonCount) and not _internal:
            logger.warning("Counter.record() is deprecated in favor of "
                           "Counter.inc() and will be removed in a future "
                           "release. Please use Counter.inc() instead.")

        if tags is not None:
            for val in tags.values():
                if not isinstance(val, str):
                    raise TypeError(
                        f"Tag values must be str, got {type(val)}.")

        final_tags = {}
        tags_copy = tags.copy() if tags else {}
        for tag_key in self._tag_keys:
            # Prefer passed tags over default tags.
            if tags is not None and tag_key in tags:
                final_tags[tag_key] = tags_copy.pop(tag_key)
            elif tag_key in self._default_tags:
                final_tags[tag_key] = self._default_tags[tag_key]
            else:
                raise ValueError(f"Missing value for tag key {tag_key}.")

        if len(tags_copy) > 0:
            raise ValueError(
                f"Unrecognized tag keys: {list(tags_copy.keys())}.")

        self._metric.record(value, tags=final_tags)

    @property
    def info(self) -> Dict[str, Any]:
        """Return the information of this metric.

        Example:
            >>> counter = Counter("name", description="desc")
                print(counter.info)
                \"""
                {
                    "name": "name",
                    "description": "desc"
                    "tag_keys": ("ray.key")
                    "default_tags": {"ray.key": "abc"}
                }
                \"""
        """
        return {
            "name": self._name,
            "description": self._description,
            "tag_keys": self._tag_keys,
            "default_tags": self._default_tags
        }


class Counter(Metric):
    """A cumulative metric that is monotonically increasing.

    This corresponds to Prometheus' counter metric:
    https://prometheus.io/docs/concepts/metric_types/#counter

    Args:
        name(str): Name of the metric.
        description(str): Description of the metric.
        tag_keys(tuple): Tag keys of the metric.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 tag_keys: Optional[Tuple[str]] = None):
        super().__init__(name, description, tag_keys)
        self._metric = CythonCount(self._name, self._description,
                                   self._tag_keys)

    def __reduce__(self):
        deserializer = Count
        serialized_data = (self._name, self._description, self._tag_keys)
        return deserializer, serialized_data

    def inc(self, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment the counter by `value` (defaults to 1).

        Args:
            value(int, float): Value to increment the counter by (default=1).
            tags(Dict[str, str]): Tags to set or override for this counter.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"value must be int or float, got {type(value)}.")
        if value <= 0:
            raise ValueError(f"value must be >0, got {value}")

        self.record(value, tags=tags, _internal=True)


class Count(Counter):
    """The count of the number of metric points.

    This corresponds to Prometheus' 'Count' metric.

    This class is DEPRECATED, please use ray.util.metrics.Counter instead.

    Args:
        name(str): Name of the metric.
        description(str): Description of the metric.
        tag_keys(tuple): Tag keys of the metric.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 tag_keys: Optional[Tuple[str]] = None):
        logger.warning(
            "`metrics.Count` has been renamed to `metrics.Counter`. "
            "`metrics.Count` will be removed in a future release.")
        super().__init__(name, description, tag_keys)


class Histogram(Metric):
    """Histogram distribution of metric points.

    This is corresponding to Prometheus' Histogram metric.
    Recording metrics with histogram will enable you to import
    min, mean, max, 25, 50, 95, 99 percentile latency.

    Args:
        name(str): Name of the metric.
        description(str): Description of the metric.
        boundaries(list): Boundaries of histogram buckets.
        tag_keys(tuple): Tag keys of the metric.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 boundaries: List[float] = None,
                 tag_keys: Optional[Tuple[str]] = None):
        super().__init__(name, description, tag_keys)
        if boundaries is None or len(boundaries) == 0:
            raise ValueError(
                "boundaries argument should be provided when using the "
                "Histogram class. e.g., Histogram(boundaries=[1.0, 2.0])")
        self.boundaries = boundaries
        self._metric = CythonHistogram(self._name, self._description,
                                       self.boundaries, self._tag_keys)

    def __reduce__(self):
        deserializer = Histogram
        serialized_data = (self._name, self._description, self.boundaries,
                           self._tag_keys)
        return deserializer, serialized_data

    @property
    def info(self):
        """Return information about histogram metric."""
        info = super().info
        info.update({"boundaries": self.boundaries})
        return info


class Gauge(Metric):
    """Gauge Keeps the last recorded value, drops everything before.

    This is corresponding to Prometheus' Gauge metric.

    Args:
        name(str): Name of the metric.
        description(str): Description of the metric.
        tag_keys(tuple): Tag keys of the metric.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 tag_keys: Optional[Tuple[str]] = None):
        super().__init__(name, description, tag_keys)
        self._metric = CythonGauge(self._name, self._description,
                                   self._tag_keys)

    def __reduce__(self):
        deserializer = Gauge
        serialized_data = (self._name, self._description, self._tag_keys)
        return deserializer, serialized_data


__all__ = [
    "Counter",
    "Histogram",
    "Gauge",
]
