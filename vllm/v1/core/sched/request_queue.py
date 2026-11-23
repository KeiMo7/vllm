# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.core.sched.fair_share import TenantAccountant
from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    FAIRSHARE = "fairshare"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class FairShareRequestQueue(RequestQueue):
    """A weighted-fair-share queue across tenants.

    Holds one FIFO per tenant. The next request is chosen by:

    1. Effective priority band of each tenant's head request. Bands improve
       with waiting time (`priority_aging_seconds`), so a low-priority
       request cannot be delayed indefinitely.
    2. Within the best band, tenants below their reserved share go first.
    3. Otherwise the tenant with the lowest virtual time (most under its
       weighted share), ties broken by arrival time then request id.

    Every input (arrival times, virtual times, request ids) derives from the
    request stream itself, so the same arrival sequence always yields the
    same pop order.
    """

    def __init__(self, accountant: TenantAccountant) -> None:
        self._accountant = accountant
        self._queues: dict[str, deque[Request]] = {}
        self._len = 0
        self._max_arrival_time = 0.0

    def _tenant_queue(self, request: Request) -> deque[Request]:
        tenant = self._accountant.tenant_of(request)
        queue = self._queues.get(tenant)
        if queue is None:
            queue = self._queues[tenant] = deque()
        return queue

    def _effective_priority(self, request: Request) -> int:
        aging = self._accountant.priority_aging_seconds
        if aging <= 0:
            return request.priority
        waited = self._max_arrival_time - request.arrival_time
        if waited <= 0:
            return request.priority
        return request.priority - int(waited // aging)

    def _select_tenant(self) -> str:
        heads = {tenant: queue[0] for tenant, queue in self._queues.items() if queue}
        if not heads:
            raise IndexError("select from an empty queue")
        best_band = min(self._effective_priority(r) for r in heads.values())
        pool = {
            tenant: request
            for tenant, request in heads.items()
            if self._effective_priority(request) == best_band
        }
        under_reserved = [
            tenant for tenant in pool if self._accountant.is_under_reserved(tenant)
        ]
        if under_reserved:
            pool = {tenant: pool[tenant] for tenant in under_reserved}
        accountant = self._accountant
        return min(
            pool.items(),
            key=lambda item: (
                accountant.virtual_time(item[0]),
                item[1].arrival_time,
                item[1].request_id,
            ),
        )[0]

    def add_request(self, request: Request) -> None:
        """Add a request to the back of its tenant's queue."""
        self._tenant_queue(request).append(request)
        self._len += 1
        self._max_arrival_time = max(self._max_arrival_time, request.arrival_time)

    def pop_request(self) -> Request:
        """Pop the request of the most under-served tenant."""
        queue = self._queues[self._select_tenant()]
        request = queue.popleft()
        self._len -= 1
        return request

    def peek_request(self) -> Request:
        """Peek at the next request without removing it."""
        return self._queues[self._select_tenant()][0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of its tenant's queue.

        Used for preempted and skipped requests, which keep their original
        arrival time and therefore their aging and tie-break position."""
        self._tenant_queue(request).appendleft(request)
        self._len += 1

    def prepend_requests(self, requests: Iterable[Request]) -> None:
        """Prepend all requests, preserving their relative order."""
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        queue = self._queues[self._accountant.tenant_of(request)]
        queue.remove(request)
        self._len -= 1

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        to_remove = requests if isinstance(requests, set) else set(requests)
        removed = 0
        for tenant in list(self._queues):
            queue = self._queues[tenant]
            kept = [r for r in queue if r not in to_remove]
            removed += len(queue) - len(kept)
            if len(kept) != len(queue):
                self._queues[tenant] = deque(kept)
        self._len -= removed

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return self._len > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return self._len

    def __iter__(self) -> Iterator[Request]:
        """Iterate in pop order, simulated on a snapshot.

        Virtual times do not change while iterating (they advance on
        charging, not on pops), so a one-cost-per-request advance is
        simulated to interleave tenants the way admissions would."""
        queues = {tenant: deque(queue) for tenant, queue in self._queues.items()}
        virtual_times: dict[str, float] = {}
        accountant = self._accountant
        while True:
            heads = {tenant: queue[0] for tenant, queue in queues.items() if queue}
            if not heads:
                return
            best_band = min(self._effective_priority(r) for r in heads.values())
            pool = {
                tenant: request
                for tenant, request in heads.items()
                if self._effective_priority(request) == best_band
            }
            tenant, request = min(
                pool.items(),
                key=lambda item: (
                    virtual_times.get(item[0], accountant.virtual_time(item[0])),
                    item[1].arrival_time,
                    item[1].request_id,
                ),
            )
            virtual_times[tenant] = virtual_times.get(
                tenant, accountant.virtual_time(tenant)
            ) + 1.0 / accountant.weight_of(tenant)
            yield request
            queues[tenant].popleft()


def create_request_queue(
    policy: SchedulingPolicy,
    fair_share: TenantAccountant | None = None,
) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.FAIRSHARE:
        assert fair_share is not None, "fairshare policy requires an accountant"
        return FairShareRequestQueue(fair_share)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
