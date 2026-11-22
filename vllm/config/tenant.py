# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tenant fair-share scheduling configuration.

Tenant identity always comes from a validated identifier carried by the
request itself (an OpenAI `X-Vllm-Tenant-Id` header, or the `tenant` argument
of the offline API); it is never derived from the request id or from the order
in which requests arrive. A request whose identifier is missing, malformed, or
not part of the configured tenant set is served from the default group, and
that decision is logged.
"""

import math
import re

from pydantic import Field, model_validator

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

# Tenant ids come from request headers, so keep them short and boring. They are
# the only request-derived string this feature writes to logs or metrics.
TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

# The default group is an internal bucket, not a tenant a request may claim.
DEFAULT_GROUP = "__default__"

# Accepts a configured identifier or the reserved default-group name.
_GROUP_PATTERN = re.compile(r"^(?:__default__|[A-Za-z0-9][A-Za-z0-9._:-]{0,63})$")

# The smallest relative share difference the ledger can represent and measure.
# A tolerance below this would ask for a precision the scheduler cannot honor,
# and would make every comparison fail.
MIN_SHARE_TOLERANCE = 1e-6


@config
class TenantShare:
    """The configured share of one tenant."""

    name: str
    """Tenant identifier as carried by requests. Must match
    `[A-Za-z0-9][A-Za-z0-9._:-]{0,63}` and be unique."""

    weight: float = Field(default=1.0, gt=0)
    """Relative share of the cluster. Shares are proportional to weights among
    the tenants that currently have work, so an idle tenant's share is lent to
    the others."""

    max_running_reqs: int | None = Field(default=None, ge=1)
    """Optional quota on concurrently running requests for this tenant. `None`
    means the tenant is limited only by the global running limit."""


@config
class TenantConfig:
    """Per-tenant fair-share scheduling.

    Disabled by default. While disabled, or when fewer than two tenants are
    configured, the scheduler keeps its original queues and code path.
    """

    enabled: bool = False
    """Turn on per-tenant fair-share admission, borrowing, and reclamation."""

    tenants: list[TenantShare] = Field(default_factory=list)
    """Configured tenants. When this list is non-empty it is also the set of
    accepted identifiers: a request naming any other tenant is served from the
    default group and the mismatch is logged."""

    default_group: str = DEFAULT_GROUP
    """Bucket for requests that do not declare a tenant, or whose identifier
    is not accepted. It is served but excluded from the cross-tenant share
    accounting, so a deployment with no declared tenants behaves exactly as
    before."""

    share_tolerance: float = Field(default=0.05, gt=0)
    """Maximum relative difference in served tokens between two active tenants
    that the scheduler accepts. Must be at least
    `MIN_SHARE_TOLERANCE`."""

    reclaim_max_steps: int = Field(default=8, ge=1)
    """Upper bound, in scheduling steps, on how long a tenant whose share was
    borrowed waits before the scheduler preempts the borrower to return it.
    It is also the cap on the catch-up burst a returning tenant may receive."""

    max_preemptions_per_step: int = Field(default=1, ge=1)
    """Cap on fairness-driven preemptions in a single scheduling step."""

    @model_validator(mode="after")
    def validate_tenants(self) -> "TenantConfig":
        if not _GROUP_PATTERN.fullmatch(self.default_group):
            raise ValueError(
                "tenants.default_group must be __default__ or match "
                "[A-Za-z0-9][A-Za-z0-9._:-]{0,63}, got "
                f"{self.default_group!r}"
            )

        seen: set[str] = set()
        for index, share in enumerate(self.tenants):
            field = f"tenants[{index}].name"
            if not TENANT_ID_PATTERN.fullmatch(share.name):
                raise ValueError(
                    f"{field} (tenant {share.name!r}) must match "
                    "[A-Za-z0-9][A-Za-z0-9._:-]{0,63}"
                )
            if share.name == self.default_group:
                raise ValueError(
                    f"{field} (tenant {share.name!r}) collides with "
                    "tenants.default_group"
                )
            if share.name in seen:
                raise ValueError(
                    f"{field} (tenant {share.name!r}) is declared more than "
                    "once; tenant identifiers must be unique"
                )
            seen.add(share.name)
            weight_field = f"tenants[{index}].weight"
            if not math.isfinite(share.weight) or share.weight <= 0:
                raise ValueError(
                    f"{weight_field} (tenant {share.name!r}) must be a finite "
                    f"positive number, got {share.weight!r}"
                )

        if not math.isfinite(self.share_tolerance):
            raise ValueError(
                f"tenants.share_tolerance must be finite, got {self.share_tolerance!r}"
            )
        if self.share_tolerance < MIN_SHARE_TOLERANCE:
            raise ValueError(
                "tenants.share_tolerance must be at least "
                f"{MIN_SHARE_TOLERANCE} (smaller values are below the "
                f"ledger's representable precision), got {self.share_tolerance!r}"
            )
        return self

    @property
    def num_configured_tenants(self) -> int:
        return len(self.tenants)

    def weight_of(self, tenant: str) -> float:
        for share in self.tenants:
            if share.name == tenant:
                return share.weight
        return 1.0

    def quota_of(self, tenant: str) -> int | None:
        for share in self.tenants:
            if share.name == tenant:
                return share.max_running_reqs
        return None

    def is_configured(self, tenant: str) -> bool:
        return any(share.name == tenant for share in self.tenants)


def validate_tenant_id(value: object) -> str | None:
    """Validate a request-carried tenant identifier.

    Returns the normalized identifier, or `None` when the value cannot be
    accepted. Callers route rejected values to the default group.
    """
    if not isinstance(value, str):
        return None
    if not TENANT_ID_PATTERN.fullmatch(value):
        return None
    return value


def resolve_tenant(
    declared: object,
    config: TenantConfig,
    *,
    warn_prefix: str = "request",
) -> tuple[str, bool]:
    """Resolve the tenant that owns a request.

    Returns `(tenant, accepted)`. `accepted` is False when the request was
    routed to the default group because its identifier was missing, malformed,
    or not configured; callers log that once per distinct identifier.
    """
    if declared is None or (isinstance(declared, str) and declared == ""):
        return config.default_group, False
    if declared == config.default_group:
        # Already resolved to the internal bucket (e.g. a request re-queued
        # after preemption). Accept it without trying to validate the name,
        # which is intentionally not a claimable identifier.
        return config.default_group, True
    tenant = validate_tenant_id(declared)
    if tenant is None:
        return config.default_group, False
    if config.tenants and not config.is_configured(tenant):
        return config.default_group, False
    return tenant, True
