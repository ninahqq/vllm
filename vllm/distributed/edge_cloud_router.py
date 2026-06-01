"""Edge-Cloud request router for multi-cloud-device inference.

This module provides request-level routing from Edge to multiple independent
Cloud devices. Each Cloud device holds a complete model replica, and the Edge
node routes requests to different Cloud devices based on load balancing.

Key invariant: once a request is bound to a Cloud device, all subsequent
tokens for that request must go to the same device (because KV Cache resides
on that device).
"""

from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import logger


@dataclass
class EdgeCloudRouter:
    """Routes requests from Edge to multiple independent Cloud devices.

    Args:
        num_cloud_devices: Number of independent Cloud devices.
        strategy: Load balancing strategy. One of "round_robin", "least_loaded".
    """

    num_cloud_devices: int
    strategy: str = "round_robin"

    # req_id -> cloud_device_id mapping
    req_to_cloud: dict[str, int] = field(default_factory=dict)
    # cloud_device_id -> number of active requests
    cloud_loads: list[int] = field(default_factory=list)
    # round-robin counter
    _rr_counter: int = field(default=0, repr=False)

    def __post_init__(self):
        if self.num_cloud_devices <= 0:
            raise ValueError(
                f"num_cloud_devices must be positive, got {self.num_cloud_devices}"
            )
        if self.strategy not in ("round_robin", "least_loaded"):
            raise ValueError(
                f"strategy must be 'round_robin' or 'least_loaded', "
                f"got {self.strategy!r}"
            )
        self.cloud_loads = [0] * self.num_cloud_devices

    def route(self, req_id: str) -> int:
        """Route a request to a Cloud device.

        If the request has already been bound, return the bound device.
        Otherwise, select a new device based on the strategy.
        """
        if req_id in self.req_to_cloud:
            return self.req_to_cloud[req_id]

        cloud_id = self._select_cloud()
        self.req_to_cloud[req_id] = cloud_id
        self.cloud_loads[cloud_id] += 1
        logger.debug(
            "EdgeCloudRouter: new request %s bound to Cloud device %d "
            "(strategy=%s, loads=%s)",
            req_id,
            cloud_id,
            self.strategy,
            self.cloud_loads,
        )
        return cloud_id

    def get_cloud(self, req_id: str) -> Optional[int]:
        """Get the Cloud device bound to a request."""
        return self.req_to_cloud.get(req_id)

    def finish(self, req_id: str) -> None:
        """Release the binding when a request finishes."""
        if req_id in self.req_to_cloud:
            cloud_id = self.req_to_cloud.pop(req_id)
            self.cloud_loads[cloud_id] = max(0, self.cloud_loads[cloud_id] - 1)
            logger.debug(
                "EdgeCloudRouter: request %s finished, released from Cloud "
                "device %d (loads=%s)",
                req_id,
                cloud_id,
                self.cloud_loads,
            )

    def _select_cloud(self) -> int:
        """Select a Cloud device based on the routing strategy."""
        if self.strategy == "round_robin":
            cloud_id = self._rr_counter % self.num_cloud_devices
            self._rr_counter += 1
            return cloud_id
        elif self.strategy == "least_loaded":
            return min(
                range(self.num_cloud_devices),
                key=lambda i: self.cloud_loads[i],
            )
        else:
            # Should never reach here due to validation in __post_init__
            raise ValueError(f"Unknown strategy: {self.strategy}")
