# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-model edge-cloud multiproc executor.

This module provides :class:`SharedModelMultiprocExecutor` and
:class:`SharedModelWorkerProc`, the executor / worker-proc pair used
on the **edge side** of the shared-model edge-cloud topology (where
multiple DP ranks on the edge share a single ``nn.Module`` in NPU
memory).

The cloud side keeps using the standard
:class:`vllm.v1.executor.multiproc_executor.MultiprocExecutor`; this
module only owns the edge side.
"""

from __future__ import annotations

import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from copy import deepcopy
from functools import cached_property, partial
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any

import cloudpickle

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle,
    MessageQueue,
)
from vllm.distributed.parallel_state import get_inner_dp_world_group_k
from vllm.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.utils import numa_utils
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    set_multiprocessing_worker_envs,
)
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


# Duck-typed detection of :class:`_BatchedExecuteMarker` (defined in
# vllm_ascend) so the upstream busy_loop does not need to import
# vllm_ascend at module load time. The marker carries a ``bundle``
# attribute (set in
# :meth:`SharedModelEdgeWorker.execute_model_batched_pre`) and a
# ``worker`` attribute (the
# :class:`SharedModelEdgeWorker` that produced it). It is also an
# :class:`AsyncModelRunnerOutput` (so the existing
# ``_pending_deferred`` machinery picks it up), but the
# ``getattr(..., "bundle", None) is not None`` test distinguishes
# batched markers from the legacy
# :class:`DeferredExecutePostprocess` returned by the original
# ``execute_model`` path.
def _is_batched_execute_marker(obj: Any) -> bool:
    return (getattr(obj, "bundle", None) is not None
            and getattr(obj, "worker", None) is not None)


# ---------------------------------------------------------------------------
# SharedModelWorkerProc
# ---------------------------------------------------------------------------
class SharedModelWorkerProc:
    """The single ``WorkerProc`` that hosts all DP-rank edge virtual
    workers.

    It implements the same protocol as the upstream
    :class:`vllm.v1.executor.multiproc_executor.WorkerProc` (i.e. is
    duck-type compatible) but **does not** inherit from it. The
    upstream class is left untouched. All upstream machinery is
    reused except for the message queues and the busy loop, which
    have to poll N MQs (one per edge executor) instead of blocking
    on a single MQ.

    MQ construction follows the upstream pattern exactly (per
    dp_rank ``k``):

    * ``self.rpc_broadcast_mqs[k]`` is the **edge-broadcast reader**
      — the return value of
      ``get_inner_dp_world_group_k(k).create_mq_broadcaster(
          external_writer_handle=shared_broadcast_handles[k],
          blocking=False)``.
      The edge executor_k is the external writer (its handle is
      passed via ``external_writer_handle``).

    * ``self.response_mqs[k]`` and
      ``self.peer_response_handles[k]`` come from
      ``get_inner_dp_world_group_k(k).create_single_reader_mq_broadcasters(
          reader_rank_in_group=0, vllm_config=vllm_config)`` —
      the return is a ``(self_response_mq, [peer_handles...])``
      tuple, exactly like the upstream ``WorkerProc``. The first
      element is the **own response writer MQ**
      (``MessageQueue(n_reader=1, n_local_reader=1)`` on this
      process) whose reader is the matching edge executor. The
      second element is the list of c cloud workers' writer
      handles, which the executor attaches as readers (mirroring
      the upstream pattern).
    """

    READY_STR = "READY"

    # Methods whose dispatch gates a per-DP polling round in the busy loop.
    # A dispatched DP is paused for the remainder of the current pass so it
    # cannot monopolize the shared model. At the end of the pass, all
    # currently available DP work is processed as one opportunistic batch.
    #
    # ``execute_model`` and ``execute_dummy_batch`` are the only
    # engine-driven per-step methods that must yield after one dispatch.
    # Independent EngineCores are not lockstep: an idle DP submits no work,
    # so requiring every DP to participate would deadlock an active peer.
    #
    # Note that the pause is *global* per dp_rank — i.e. while a
    # dp_rank is paused, the busy loop will not dequeue ANY rpc
    # (including non-SYNC methods such as ``add_lora``) from that
    # MQ. Work queued behind a SYNC method is picked up on the next pass,
    # after the current available batch has been drained.
    SYNC_METHODS: frozenset[str] = frozenset(
        {"execute_model", "execute_dummy_batch"})

    # ------------------------------------------------------------------ init
    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,  # noqa: ARG002 - protocol parity
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        shared_broadcast_handles: dict[int, Handle] | None = None,
        global_dp_ranks: tuple[int, ...] | None = None,
    ) -> None:
        # Stash BEFORE the worker is constructed. The worker class's
        # __init__ calls init_distributed_environment which builds
        # _INNER_DP_WORLD_EDGE_LIST (the edge-side per-dp_rank gloo
        # groups we need to construct the readers/writers with).
        self.rank = rank
        self._shared_broadcast_handles: dict[int, Handle] = (
            shared_broadcast_handles or {})
        self.global_dp_ranks = (
            global_dp_ranks
            or tuple(range(vllm_config.parallel_config.data_parallel_size))
        )
        self._global_to_local_dp_rank = {
            global_dp_rank: local_dp_rank
            for local_dp_rank, global_dp_rank in enumerate(self.global_dp_ranks)
        }

        # Construct one virtual worker per DP rank hosted by this physical edge
        # group. ``local_rank`` is group-local; ``global_dp_rank`` is retained
        # separately for diagnostics and control-plane routing.
        # Each virtual worker has its own WorkerWrapperBase, with
        # ``local_rank=k`` matching its dp_rank. The configured
        # worker class (SharedModelEdgeWorker) does the in-process
        # sharing of model weights.
        group_size = len(self.global_dp_ranks)
        self.worker: list[WorkerWrapperBase] = []
        for local_dp_rank, global_dp_rank in enumerate(self.global_dp_ranks):
            # A model runner keeps a reference to ParallelConfig and consults
            # its DP rank throughout request preparation/sampling.  Virtual
            # workers must therefore not share the leader EngineCore's mutable
            # VllmConfig: doing so makes every runner identify as the group
            # leader (DP0 for group 0, DP2 for group 1), which can route a
            # request's new and cached scheduler steps through different
            # virtual state dictionaries.
            worker_config = deepcopy(vllm_config)
            worker_parallel_config = worker_config.parallel_config
            worker_parallel_config.data_parallel_rank = global_dp_rank
            worker_parallel_config.data_parallel_rank_local = local_dp_rank
            worker_parallel_config.data_parallel_index = global_dp_rank

            wrapper = WorkerWrapperBase(
                rpc_rank=local_dp_rank,
                # Each EngineCore's KV-config list is ordered by its response
                # MQs: the shared edge worker is always entry 0, followed by
                # that DP rank's cloud workers. ``rank`` is the physical edge
                # group id, not an index into this per-EngineCore list; using
                # it makes edge group 1 consume a cloud-only KV config.
                global_rank=0,
            )
            # The upstream WorkerProc.__init__ does NOT include
            # shared_broadcast_handles in all_kwargs; that is the
            # one divergence we need. The configured worker class
            # can read it through its own __init__ kwargs if it
            # wants to.
            all_kwargs: list[dict] = [
                {}
                for _ in range(
                    max(vllm_config.parallel_config.world_size, group_size)
                )
            ]
            all_kwargs[local_dp_rank] = {
                "vllm_config": worker_config,
                "local_rank": local_dp_rank,
                "rank": rank,
                "distributed_init_method": distributed_init_method,
                "is_driver_worker": is_driver_worker,
                "shared_worker_lock": shared_worker_lock,
                "global_dp_rank": global_dp_rank,
            }
            wrapper.init_worker(all_kwargs)
            self.worker.append(wrapper)

        # Initialise each virtual worker's device / model.
        for k, wrapper in enumerate(self.worker):
            wrapper.init_device()
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                wrapper.elastic_ep_execute("load_model")
            else:
                wrapper.load_model()

        # Set block size based on the attention backends
        current_platform.update_block_size_for_backend(vllm_config)

        # Each virtual DP owns an independent output queue and materialization
        # thread. A pending NPU event on one DP must not block the shared worker
        # dispatch loop or the response stream of another DP.
        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        self.async_output_queues: list[queue.Queue[Any]] = []
        self.async_output_copy_threads: list[Thread] = []
        self._async_output_stop = object()
        self._async_output_errors: list[BaseException | None] = [None] * group_size

        # Message queue setup: N (one per dp_rank) broadcast
        # readers + N response writer/peer pairs. The standard
        # _init_message_queues is not used.
        self._init_message_queues(input_shm_handle, vllm_config)
        if self.use_async_scheduling:
            self.async_output_queues = [queue.Queue() for _ in range(group_size)]
            for dp_rank in range(group_size):
                output_thread = Thread(
                    target=self.async_output_busy_loop,
                    args=(dp_rank,),
                    daemon=True,
                    name=f"SharedEdgeAsyncOutput-DP{dp_rank}",
                )
                output_thread.start()
                self.async_output_copy_threads.append(output_thread)

        # Pending ``AsyncModelRunnerOutput`` markers (i.e. the
        # ``DeferredExecutePostprocess`` instances returned by
        # :meth:`SharedModelEdgeWorker.execute_model` for the
        # tail recv + tail forward) keyed by dp_rank. Populated
        # by ``_dispatch`` and drained by ``worker_busy_loop``
        # at the end of each round, so per-dp_rank tail
        # processing happens in lockstep. The dict is ordered
        # by insertion (Python 3.7+) so the round's tail
        # processing happens in dispatch order.
        self._pending_deferred: dict[int, AsyncModelRunnerOutput] = {}

        # Batched-compute round state (Step 11). The dispatch
        # phase populates ``_round_bundles[k]`` (one bundle per
        # dp_rank carrying the per-dp_rank preprocess state from
        # ``NPUModelRunner.execute_model_pre``); the drain phase
        # populates ``_round_intermediates[k]`` with the
        # ``IntermediateTensors`` received back from the cloud.
        # Both are cleared at end-of-round. The types are loaded
        # lazily inside the methods that touch them to avoid an
        # import cycle with vllm_ascend at module import time.
        self._round_bundles: dict[int, Any] = {}
        self._round_intermediates: dict[int, IntermediateTensors] = {}

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()
        self.is_moe = vllm_config.model_config.is_moe

    # ---------------------------------------------------- MQ setup (overridden)
    def _init_message_queues(
        self,
        input_shm_handle: Handle,  # noqa: ARG002
        vllm_config: VllmConfig,
    ) -> None:
        """Construct the dp_size reader/writer MQ pairs.

        Following the upstream pattern, both MQs are produced by
        ``GroupCoordinator`` helpers (``create_mq_broadcaster`` and
        ``create_single_reader_mq_broadcasters``). The only
        difference is that we do this **once per dp_rank** instead
        of once per process.

        Blocking is ``False`` everywhere; the readiness barrier is
        enforced separately via ``wait_until_ready`` in
        ``worker_main``.
        """
        group_size = len(self.global_dp_ranks)

        # 1) attach to N edge-broadcast MQs as reader (one per
        #    dp_rank). The edge executor's broadcast handle is the
        #    external writer.
        self.rpc_broadcast_mqs = [
            get_inner_dp_world_group_k(k).create_mq_broadcaster(
                external_writer_handle=(
                    self._shared_broadcast_handles[global_dp_rank]
                ),
                blocking=False,
            )
            for k, global_dp_rank in enumerate(self.global_dp_ranks)
        ]

        # 2) build the N response MQ pairs (one per dp_rank):
        #    own response_mq (writer, reader = edge executor_k) +
        #    peer_handles (c cloud workers' writer handles). The
        #    return shape matches the upstream ``WorkerProc``:
        #    ``(self_response_mq, [peer_handles...])``.
        self.response_mqs = []
        self.peer_response_handles = []
        for k in range(group_size):
            response_mq, peer_handles = (
                get_inner_dp_world_group_k(k)
                .create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0,
                    blocking=False,
                    vllm_config=vllm_config,
                ))
            self.response_mqs.append(response_mq)
            self.peer_response_handles.append(peer_handles or [])

    # --------------------------------------------------- busy_loop (overridden)
    def worker_busy_loop(self) -> None:
        """Poll the dp_size edge-broadcast MQs and dispatch to the
        matching virtual worker.

        The standard upstream loop blocks on a single
        ``self.rpc_broadcast_mq.dequeue(indefinite=True)``; the
        shared model design needs a multi-MQ round-robin because
        :class:`MessageQueue` has no multi-MQ select primitive.

        Cross-DP pacing
        --------------------
        The shared model runner polls every virtual DP once per pass and
        opportunistically batches the work already available in that pass:

        * On every dispatch whose method is in
          :attr:`SYNC_METHODS`, the dispatching dp_rank is
          marked *paused* and the busy loop will not dequeue
          from its MQ for the rest of the current pass.
        * At the end of the pass, any paused DPs form the current batch.
          Idle DPs are not required to participate.
        * KV initialization remains a full-group operation and waits until
          every virtual DP has registered its cache configuration.
        * The pause is *global* per dp_rank — non-SYNC methods
          (e.g. ``add_lora``) queued behind a SYNC method on
          the same MQ are picked up on the next pass.

        Batch postprocess of ``execute_model`` results
        ----------------------------------------------
        ``SharedModelEdgeWorker.execute_model`` does the head
        forward + PP send synchronously and returns the tail
        recv + tail forward wrapped in a
        :class:`DeferredExecutePostprocess` marker — an
        :class:`AsyncModelRunnerOutput` subclass that is also
        callable. ``_dispatch`` detects the marker via
        ``isinstance(output, AsyncModelRunnerOutput)`` and
        stores it in ``self._pending_deferred``; when the
        round barrier is reached at the end of the pass, the
        busy loop calls each marker first (the call invokes
        ``__call__``, which runs the deferred tail processing
        and returns the raw postprocess result — no extra
        type check), then routes the result to the matching
        response MQ via ``handle_output`` →
        ``enqueue_output``. The downstream
        ``enqueue_output`` does its own
        ``isinstance(output, AsyncModelRunnerOutput)`` /
        ``get_output()`` unwrap, in case the postprocess
        itself returned an async output. The
        ``DeferredExecutePostprocess.get_output`` method (an
        alternative entry point that does one more type
        check) is reserved for direct callers that bypass
        ``enqueue_output``.
        """
        assert self.rpc_broadcast_mqs, (
            "rpc_broadcast_mqs must be initialised before busy loop")

        dp_size = len(self.rpc_broadcast_mqs)
        paused = [False] * dp_size

        init_kv_cache = False
        while True:
            dispatched = False
            for k, mq in enumerate(self.rpc_broadcast_mqs):
                if paused[k]:
                    continue
                try:
                    # ``dequeue`` is the upstream high-level API
                    # used by the standard ``MultiprocExecutor``
                    # busy loop. Using the low-level
                    # ``acquire_read`` would bypass the
                    # status / timeout semantics that the rest of
                    # vLLM's executor protocol depends on.
                    method, args, kwargs, output_rank = mq.dequeue(
                        timeout=0.001)
                except TimeoutError:
                    continue
                # outer enumerate index k is the dp_rank
                self._dispatch(k, method, args, kwargs, output_rank)
                dispatched = True
                # Round barrier: pause this dp_rank's MQ intake
                # for the rest of the current round. The check
                # uses ``in`` on a ``frozenset[str]`` so a
                # ``bytes``-method (cloudpickled callable) just
                # does not match — only the standard engine-
                # driven methods gate the barrier.
                #
                # Exception: an ``execute_model`` call with
                # ``num_scheduled_tokens == 0`` is an "empty"
                # scheduler step — the model runner returns
                # without doing a real forward, no PP send /
                # tail recv happens, and the result is sent
                # back immediately (no ``DeferredExecutePostprocess``
                # marker either). Such a call does not need to
                # gate the round barrier, so we let the dp_rank
                # keep dispatching instead of pausing it. The
                # first positional arg is the
                # ``SchedulerOutput``; the engine still calls
                # ``execute_model`` once per scheduler step
                # regardless of the batch size, so this just
                # means the pause is only triggered by the
                # "real" forward calls.
                is_empty_execute = (
                    method == "execute_model"
                    and args
                    and getattr(args[0],
                                "total_num_scheduled_tokens", 0) == 0)
                if (method in self.SYNC_METHODS
                        and not is_empty_execute):
                    paused[k] = True
                if method == 'initialize_from_config':
                    paused[k] = True
                    init_kv_cache = True
            # End-of-round: after polling every virtual DP once, drain the
            # currently available execute set as soon as at least one DP has
            # paused. This opportunistically batches DP requests that were
            # already queued in the same polling pass, while allowing a busy
            # DP to make progress when another DP in the group is idle.
            #
            # Waiting for ``all(paused)`` is invalid for independently
            # scheduled EngineCores: an idle peer does not issue
            # ``execute_model``, so the active DP's immediately-following
            # ``sample_tokens`` RPC would remain behind the pause forever.
            # KV initialization is the exception and still requires every
            # virtual DP to register before the shared allocation is built.
            #
            # Unpause everyone for the next round AND — crucially — drain
            # the pending ``AsyncModelRunnerOutput`` markers that
            # ``_dispatch`` accumulated during the round. For each
            # marker: if it is *callable* (i.e. a
            # ``DeferredExecutePostprocess``), call it first to
            # run the deferred tail recv + tail forward and get
            # the raw postprocess result; the result is then
            # routed to the matching response MQ via
            # ``handle_output`` (which goes through
            # ``enqueue_output`` and unwraps the result via
            # ``isinstance(output, AsyncModelRunnerOutput)`` /
            # ``get_output()`` if the postprocess itself
            # returned an async output). Exceptions raised by
            # the postprocess are caught and converted to a
            # FAILURE response on the matching response MQ so
            # the engine sees the failure rather than the busy
            # loop crashing. The insertion-ordered iteration
            # over ``self._pending_deferred`` preserves
            # dispatch order.
            round_ready = (
                all(paused) if init_kv_cache else any(paused)
            )
            if round_ready:
                if init_kv_cache:
                    init_kv_cache = False
                    paused = [False] * dp_size
                    continue
                paused = [False] * dp_size
                if self._pending_deferred:
                    pending, self._pending_deferred = (
                        self._pending_deferred, {})

                    # Step 11 batched-compute path: when the
                    # ``_dispatch`` step accumulated
                    # :class:`_BatchedExecuteMarker`s, drive the
                    # batched round via the marker's class and
                    # member functions. Phase A (1× batched head +
                    # per-dp_rank PP isend + recv closure) runs
                    # first; Phase B/C (recv + 1× batched tail +
                    # per-dp_rank post_batched + handle_output)
                    # runs at the end. The marker is the single
                    # owner of all the batched logic — the
                    # busy_loop only routes the round state dicts
                    # and the response-MQ callback.
                    #
                    # Mixed-SYNC partitioning: the batched path
                    # only operates on the dp_ranks that produced
                    # batched markers in this round. Any
                    # non-batched entries in ``pending`` (legacy
                    # ``DeferredExecutePostprocess`` from the
                    # original ``execute_model`` path, used by
                    # non-batched callers) are routed through the
                    # legacy ``deferred()`` loop below — that path
                    # runs the original head + send / tail recv +
                    # tail forward end-to-end and is fully
                    # independent of the batched path. This way
                    # the batched head / tail / post only see
                    # the dp_ranks that actually want to be
                    # merged.
                    batched_dp_ranks_list = sorted(
                        k for k, d in pending.items()
                        if _is_batched_execute_marker(d))
                    batched_pending: dict[int, Any] = {
                        k: pending[k] for k in batched_dp_ranks_list
                    }
                    legacy_pending: dict[int, Any] = {
                        k: d for k, d in pending.items()
                        if k not in batched_pending
                    }
                    if batched_pending:
                        # Phase A — batched head (1× per round).
                        # ``run_batched_head`` and ``drain_batched_round``
                        # are class functions on the vllm_ascend
                        # ``_BatchedExecuteMarker``; reach them via
                        # ``type(marker)`` to avoid an upstream
                        # import of vllm_ascend.
                        marker_cls = type(batched_pending[
                            batched_dp_ranks_list[0]])
                        try:
                            marker_cls.run_batched_head(
                                batched_dp_ranks_list,
                                self._round_bundles)
                        except Exception as e:
                            if hasattr(e, "add_note"):
                                e.add_note(traceback.format_exc())
                            logger.exception(
                                "SharedModelWorkerProc hit an exception "
                                "running batched head.")
                            for k in batched_dp_ranks_list:
                                self.handle_output(k, e)
                            self._round_bundles.clear()
                            self._round_intermediates.clear()
                            marker_cls._per_dp_hidden = None
                            dispatched = True
                            continue

                        # Phase A — per-dp_rank PP isend + recv
                        # closure. Each batched marker installs its
                        # own recv closure into ``batched_pending``.
                        drive_failures: dict[int, Exception] = {}
                        for dp_rank, marker in batched_pending.items():
                            try:
                                marker.drive_batched_round(
                                    batched_pending)
                            except Exception as e:
                                if hasattr(e, "add_note"):
                                    e.add_note(traceback.format_exc())
                                logger.exception(
                                    "SharedModelWorkerProc hit an "
                                    "exception running per-dp_rank "
                                    "PP isend / recv closure on "
                                    "dp_rank=%d.", dp_rank)
                                drive_failures[dp_rank] = e

                        # Surface any per-dp_rank Phase A failure as
                        # a FAILURE response on its MQ.
                        for k, e in drive_failures.items():
                            self.handle_output(k, e)
                            self._round_bundles.pop(k, None)
                            batched_pending.pop(k, None)

                        # Phase B/C — recv + batched tail + per-dp_rank
                        # post_batched + handle_output. The marker
                        # class function drives this end-to-end,
                        # operating only on ``batched_pending``.
                        marker_cls.drain_batched_round(
                            self._round_bundles,
                            self._round_intermediates,
                            batched_pending,
                            on_dp_rank_output=self.handle_output,
                        )
                        # ``drain_batched_round`` has cleared
                        # ``_round_bundles`` / ``_round_intermediates``
                        # at end-of-round; the legacy entries have
                        # not been touched and are still in
                        # ``legacy_pending`` below.

                    # Legacy / non-batched path: each
                    # ``deferred`` is either a legacy
                    # ``DeferredExecutePostprocess`` (its
                    # ``__call__`` runs the original head + send /
                    # tail recv + tail forward end-to-end) or a
                    # raw output (passed through). The
                    # ``batched_markers`` partition above has
                    # already removed any batched entries from
                    # this loop.
                    for dp_rank, deferred in legacy_pending.items():
                        try:
                            output = deferred() if callable(deferred) else deferred
                        except Exception as e:
                            if hasattr(e, "add_note"):
                                e.add_note(traceback.format_exc())
                            logger.exception(
                                "SharedModelWorkerProc hit an exception "
                                "running deferred execute_model "
                                "postprocess on dp_rank=%d.", dp_rank)
                            output = e
                        self.handle_output(dp_rank, output)
                    dispatched = True
            if not dispatched:
                # No MQ had a message in this pass; yield to let
                # the death-pipe monitor and signal handlers run.
                time.sleep(0)

    def _dispatch(
        self,
        dp_rank: int,
        method: str,
        args: tuple,
        kwargs: dict,
        output_rank: int | None,
    ) -> None:
        """Dispatch a method call to the dp_rank-th virtual worker.

        The result is enqueued into
        ``self.response_mqs[dp_rank]`` (the own response_mq
        returned by ``create_single_reader_mq_broadcasters``) when
        ``output_rank`` matches (mirroring the upstream
        ``worker_busy_loop`` output-routing convention).

        ``output_rank`` is executor-local: rank 0 means this shared edge
        WorkerProc, regardless of its physical distributed global rank.
        In grouped-shared mode group 1 has physical rank 1, so comparing
        ``output_rank`` with ``self.rank`` would silently discard every
        DP2/DP3 response.
        """
        virtual_worker = self.worker[dp_rank]
        is_output_worker = output_rank is None or output_rank == 0
        try:
            # Step 11 batched path: route ``execute_model`` RPCs to
            # the new ``execute_model_batched_pre`` interface, which
            # only does per-dp_rank preprocess and returns a
            # ``_BatchedExecuteMarker`` (or an early-return
            # ``ModelRunnerOutput`` / ``None`` for no-work cases).
            # Workers may reject the batched path for scheduler outputs whose
            # per-request state cannot be merged. Those outputs retain the
            # original stateful ``execute_model`` head/send path; otherwise
            # the busy loop drives batched head/tail/post on the leader runner.
            if method == "execute_model" and hasattr(
                    virtual_worker.worker, "execute_model_batched_pre"):
                scheduler_output = args[0] if args else None
                should_use_batched = getattr(
                    virtual_worker.worker,
                    "should_use_batched_execute",
                    None,
                )
                # Some shared workers can batch only a subset of scheduler
                # outputs. Edge-cloud multimodal requests, for example, own
                # per-request encoder and M-RoPE state that cannot be merged
                # by the virtual-DP fast path. Keep this accelerator-specific
                # policy duck-typed in the worker implementation.
                use_batched = (
                    should_use_batched is None
                    or should_use_batched(scheduler_output)
                )
                if use_batched:
                    output = virtual_worker.execute_model_batched_pre(
                        scheduler_output)
                else:
                    logger.info_once(
                        "Shared-model batched execution is bypassed for "
                        "this scheduler output; using the virtual worker's "
                        "stateful execute_model path.")
                    output = virtual_worker.execute_model(
                        scheduler_output)
            elif isinstance(method, str):
                func = getattr(virtual_worker, method)
                output = func(*args, **kwargs)
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), virtual_worker)
                output = func(*args, **kwargs)
            else:
                output = None
        except Exception as e:
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            logger.exception(
                "SharedModelWorkerProc hit an exception on dp_rank=%d.",
                dp_rank)
            if is_output_worker:
                self.handle_output(dp_rank, e)
            return

        if is_output_worker:
            # Step 11 batched-compute path: when ``execute_model`` is
            # dispatched, the worker returns a marker that carries a
            # per-dp_rank bundle (in ``output.bundle``). Duck-typed
            # check — no vllm_ascend import needed in this upstream
            # module. The marker is stored in ``_pending_deferred``
            # like any other async marker, and its ``__call__`` (run
            # by the round barrier in :meth:`worker_busy_loop`)
            # drives the batched head / recv / tail / per-dp_rank
            # post end-to-end.
            if (method == "execute_model"
                    and getattr(output, "bundle", None) is not None):
                self._round_bundles[dp_rank] = output.bundle
                self._pending_deferred[dp_rank] = output
            elif (method == "execute_model"
                  and isinstance(output, AsyncModelRunnerOutput)):
                # Legacy ``DeferredExecutePostprocess`` from the
                # original ``execute_model`` path (kept around for
                # any non-batched callers).
                self._pending_deferred[dp_rank] = output
            else:
                self.handle_output(dp_rank, output)

    def handle_output(self, dp_rank: int, output: Any) -> None:
        """Route a worker output to the matching DP response MQ.

        With async scheduling, each virtual DP has its own FIFO output queue
        and materialization thread. Immediate and asynchronous outputs use the
        same queue so they cannot be reordered within a DP, while an NPU event
        wait on one DP cannot block another DP or the shared dispatch thread.
        """
        if not 0 <= dp_rank < len(self.response_mqs):
            raise IndexError(
                f"Invalid local dp_rank={dp_rank}; "
                f"response_mq_count={len(self.response_mqs)}"
            )
        if self.use_async_scheduling:
            output_error = self._async_output_errors[dp_rank]
            if output_error is not None:
                raise RuntimeError(
                    "Shared edge async output thread failed for "
                    f"local_dp_rank={dp_rank}"
                ) from output_error
            if len(self.async_output_queues) != len(self.response_mqs):
                raise RuntimeError(
                    "Shared edge async output queues are not initialized: "
                    f"queue_count={len(self.async_output_queues)}, "
                    f"response_mq_count={len(self.response_mqs)}"
                )
            self.async_output_queues[dp_rank].put(output)
        else:
            self.enqueue_output(dp_rank, output)

    def enqueue_output(self, dp_rank: int, output: Any) -> None:
        """Write a single response to ``response_mqs[dp_rank]``,
        wrapping it in the standard ``(status, payload)`` tuple.
        If ``output`` is an ``AsyncModelRunnerOutput`` (async
        scheduling mode), extract the real output first.
        """
        try:
            if isinstance(output, AsyncModelRunnerOutput):
                output = output.get_output()
        except Exception as e:
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            logger.exception(
                "Shared edge async output materialization failed on local_dp_rank=%d.",
                dp_rank,
            )
            output = e
        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        self.response_mqs[dp_rank].enqueue(result)

    def async_output_busy_loop(self, dp_rank: int) -> None:
        """Materialize one virtual DP's outputs in strict FIFO order."""
        try:
            virtual_workers = getattr(self, "worker", [])
            if (
                dp_rank < len(virtual_workers)
                and hasattr(virtual_workers[dp_rank], "device")
            ):
                current_platform.set_device(virtual_workers[dp_rank].device)

            output_queue = self.async_output_queues[dp_rank]
            while True:
                output = output_queue.get()
                if output is self._async_output_stop:
                    return
                self.enqueue_output(dp_rank, output)
        except BaseException as e:
            self._async_output_errors[dp_rank] = e
            logger.exception(
                "Shared edge async output thread failed for "
                "local_dp_rank=%d.",
                dp_rank,
            )

    def _stop_async_output_threads(self) -> None:
        """Drain and stop every per-DP output thread before closing MQs."""
        output_queues = getattr(self, "async_output_queues", [])
        output_threads = getattr(self, "async_output_copy_threads", [])
        for output_queue in output_queues:
            output_queue.put(self._async_output_stop)
        for dp_rank, output_thread in enumerate(output_threads):
            output_thread.join(timeout=10)
            if output_thread.is_alive():
                logger.warning(
                    "Shared edge async output thread for local_dp_rank=%d "
                    "did not stop within 10 seconds.",
                    dp_rank,
                )
        self.async_output_copy_threads = []
        self.async_output_queues = []

    # ------------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        # Stop markers are appended after pending outputs, so each DP drains
        # its own FIFO before its response MQ is closed.
        self._stop_async_output_threads()
        for mq in getattr(self, "rpc_broadcast_mqs", []) or []:
            if mq is not None:
                mq.shutdown()
        for mq in getattr(self, "response_mqs", []) or []:
            if mq is not None:
                mq.shutdown()
        self.rpc_broadcast_mqs = []
        self.response_mqs = []
        for virtual_worker in getattr(self, "worker", []) or []:
            if virtual_worker is not None:
                virtual_worker.shutdown()
        self.worker = []
        destroy_model_parallel()
        destroy_distributed_environment()

    def monitor_death_pipe(self, death_pipe, shutdown_requested):
        if death_pipe is None:
            return

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        Thread(
            target=death_pipe_monitor,
            args=(list(self.rpc_broadcast_mqs)
                  + list(self.response_mqs),),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()

    # ------------------------------------------------- static process helpers
    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # noqa: ARG001 - protocol parity
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        shared_broadcast_handles: dict[int, Handle] | None = None,
        global_dp_ranks: tuple[int, ...] | None = None,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(),
                                  death_writer.fileno()))
        process_kwargs: dict[str, Any] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": None,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            "inherited_fds":
                inherited_fds if inherited_fds is not None else [],
            "shared_broadcast_handles": shared_broadcast_handles,
            "global_dp_ranks": global_dp_ranks,
        }
        proc = context.Process(
            target=SharedModelWorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmSharedEdgeWorker-{rank}",
            daemon=True,
        )
        with numa_utils.configure_subprocess(
                vllm_config, local_rank, process_kind="worker"):
            proc.start()
        ready_writer.close()
        death_reader.close()
        return UnreadyWorkerProcHandle(proc, rank, ready_reader,
                                       death_writer)

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> dict[int, Handle]:
        """Block until the single shared edge ``WorkerProc`` becomes
        ready, then return the per-dp_rank response handles (one
        for each dp_rank this executor will read from). The dict
        shape is ``{dp_rank: (own_response_handle,
        [peer_handles...]) }``.

        The standard upstream ``wait_for_ready`` returns a list of
        ``WorkerProcHandle`` objects with ``peer_worker_response_mqs``
        as already-attached ``MessageQueue`` instances; that shape
        does not fit our design (we have dp_size×c peer handles
        distributed across dp_size per-dp_rank gloo sub-groups, and
        we want the raw handles to attach readers from per-executor
        state). This function only returns the raw handle dict so
        the executor can attach them with the right reader rank
        and dispatch policy.
        """
        assert len(unready_proc_handles) == 1, (
            "SharedModelWorkerProc only supports a single worker")
        proc_handle = unready_proc_handles[0]
        try:
            response: dict[str, Any] = proc_handle.ready_pipe.recv()
        finally:
            proc_handle.ready_pipe.close()
        if response["status"] != SharedModelWorkerProc.READY_STR:
            raise RuntimeError(
                f"Shared edge WorkerProc failed to become ready: "
                f"{response}")
        return response  # type: ignore[return-value]

    @staticmethod
    def worker_main(*args, **kwargs) -> None:
        """Worker process entry point (mirrors upstream)."""
        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "SharedModelWorkerProc handling signal %d, "
                    "raising SystemExit", signum)
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker: SharedModelWorkerProc | None = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s",
                               type(e), e)
        try:
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"SharedEdgeWorker_{rank}",
            )
            worker = SharedModelWorkerProc(*args, **kwargs)
            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # Build the ready dict in the upstream shape:
            #   "handle"                 -> this WorkerProc's own
            #                              response writer handle
            #                              (per dp_rank)
            #   "peer_response_handles"  -> c cloud workers' writer
            #                              handles (per dp_rank)
            # We use the upstream dict keys so the executor-side
            # mirror of wait_for_response_handle_ready works
            # naturally.
            ready_writer.send({
                "status":
                SharedModelWorkerProc.READY_STR,
                "response_handles": [
                    mq.export_handle() for mq in worker.response_mqs
                ],
                "peer_response_handles": worker.peer_response_handles,
            })

            # Wait for all readers to subscribe.
            for mq in worker.rpc_broadcast_mqs:
                mq.wait_until_ready()
            for mq in worker.response_mqs:
                mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop()
        except Exception:
            if ready_writer is not None:
                logger.exception("SharedModelWorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.info("SharedModelWorkerProc shutting down.")
            else:
                logger.exception("SharedModelWorkerProc failed.")
            shutdown_requested.set()
        except SystemExit as e:
            logger.warning("SharedModelWorkerProc was terminated")
            raise e
        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            if worker is not None:
                worker.shutdown()


# ---------------------------------------------------------------------------
# SharedModelMultiprocExecutor
# ---------------------------------------------------------------------------
class SharedModelMultiprocExecutor(MultiprocExecutor):
    """Edge-side executor for the shared-model edge-cloud topology.

    This class is a near-verbatim copy of
    :meth:`MultiprocExecutor._init_executor`. The only differences
    are:

    * The local world is always size 1: one edge executor process
      per DP rank, with no local workers of its own.
    * Each edge executor owns one broadcast ``MessageQueue``
      (writer, 1 reader = the shared edge ``WorkerProc``).
    * Each edge executor's ``self.response_mqs`` is a list of
      ``MessageQueue`` readers: one for the shared edge
      ``WorkerProc``'s own response MQ (for this dp_rank) plus c
      readers for the c cloud workers' response MQs (for this
      dp_rank). Both are produced by reading the raw handles from
      the worker's ready dict and attaching them locally.
    * The first DP rank in each edge group starts that group's shared
      ``WorkerProc``; the other group members exchange handles through
      the gloo store.
    """

    @cached_property
    def max_concurrent_batches(self) -> int:
        # The shared edge path is not a standard PP pipeline. Each virtual
        # runner owns a single execute_model_state slot. Async scheduling
        # supports the standard two-batch overlap; without async scheduling,
        # execution must remain strictly single-batch. In particular, do not
        # inherit ``pipeline_parallel_size`` from MultiprocExecutor here:
        # edge-cloud PP is driven explicitly by the shared worker and is not a
        # standard local pipeline whose bubbles are filled by EngineCore's
        # batch queue.
        return 2 if self.scheduler_config.async_scheduling else 1

    def _init_executor(self) -> None:
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        # The local world is always size 1: one edge executor
        # process per DP rank, with no local workers of its own.
        self._get_parallel_sizes()

        set_multiprocessing_worker_envs()

        # Each edge executor creates its own broadcast MQ (writer,
        # 1 reader = the shared edge WorkerProc).
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        mq_connect_ip = get_ip()
        dp_size = self.parallel_config.data_parallel_size
        self.edge_group_id = self.parallel_config.edge_group_id(
            self.parallel_config.data_parallel_rank)
        self.group_dp_ranks = self.parallel_config.edge_group_dp_ranks(
            self.edge_group_id)
        self.group_local_dp_rank = (
            self.parallel_config.edge_group_local_rank(
                self.parallel_config.data_parallel_rank))
        self.group_leader_dp_rank = (
            self.parallel_config.edge_group_leader_dp_rank(
                self.edge_group_id))
        logger.info(
            "SharedModel edge executor: dp_rank=%d, dp_size=%d, "
            "edge_group_id=%d, group_local_dp_rank=%d, group_dp_ranks=%s, "
            "world_size=%d, local_world_size=%d, mq_connect_ip=%s",
            self.parallel_config.data_parallel_rank,
            dp_size,
            self.edge_group_id,
            self.group_local_dp_rank,
            self.group_dp_ranks,
            self.world_size,
            self.local_world_size,
            mq_connect_ip,
        )
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            self.local_world_size,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=mq_connect_ip,
        )
        own_broadcast_handle = self.rpc_broadcast_mq.export_handle()

        # Join the edge control gloo group and use its store to exchange
        # namespaced handles. Each group leader starts one shared edge
        # WorkerProc after its own group has published all broadcast handles.
        dp_group_pg, dp_group_store = (
            self.parallel_config.stateless_init_dp_group(
                return_store=True))
        self._dp_group_pg = dp_group_pg
        self._dp_group_store = dp_group_store

        # ``unready_workers`` tracks the worker proc handles
        # produced by ``make_worker_process`` so that, if any
        # step in the init sequence below raises, we still
        # close the death pipes and ensure worker termination
        # (mirroring the upstream
        # ``MultiprocExecutor._init_executor`` cleanup pattern).
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            # Step 1: each executor publishes its own broadcast
            # handle to the gloo store.
            self._publish_broadcast_handle(own_broadcast_handle)
            # Step 2: the group leader starts its shared edge WorkerProc
            # and waits for it to become ready (returns the raw
            # handle dict). All other executors do nothing — they
            # will read the per-dp_rank handles from the gloo
            # store once the master has published them.
            ready_dict, unready_workers = (
                self._spawn_shared_edge_worker_if_master())
            # Step 3: this executor attaches to its own
            # response MQ (reader) and to the c cloud workers'
            # response MQs (readers) by reading the per-dp_rank
            # handles from the ready dict / gloo store.
            self._attach_response_mqs(ready_dict)
            # Ensure message queues are ready. Will deadlock if
            # re-ordered; must be kept consistent with the
            # ``SharedModelWorkerProc`` busy-loop entry barrier.
            # Wait for the broadcast writer (us) to be fully
            # connected to all readers (the shared edge
            # ``WorkerProc`` and the c cloud workers for this
            # dp_rank via the per-dp_rank gloo sub-group).
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for the response readers (us) to be fully
            # connected to the writers (the shared edge
            # ``WorkerProc`` and the c cloud workers).
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()
            self.futures_queue = deque[FutureWrapper]()
            self.output_rank = 0  # the edge is the only output rank
            self._post_init_executor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a
                # failure. Close death_writers first to signal
                # workers to exit, then ensure termination. This
                # mirrors the upstream ``MultiprocExecutor``
                # cleanup so the child processes do not leak if
                # init failed.
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])
                if rpc_broadcast_mq := getattr(
                        self, "rpc_broadcast_mq", None):
                    rpc_broadcast_mq.shutdown()
                    self.rpc_broadcast_mq = None
                if response_mqs := getattr(self, "response_mqs", None):
                    for mq in response_mqs:
                        mq.shutdown()
                    self.response_mqs = []

    # ----------------------------------------------- handle exchange helpers
    def _publish_broadcast_handle(self, handle: Handle) -> None:
        """Publish this executor's broadcast MQ handle to the
        edge-only gloo store under a per-dp_rank key."""
        assert self._dp_group_store is not None
        key = (
            f"shared_edge/{self.edge_group_id}/broadcast/"
            f"{self.parallel_config.data_parallel_rank}"
        )
        self._dp_group_store.set(key, pickle.dumps(handle))

    def _collect_broadcast_handles(self) -> dict[int, Handle]:
        """Group leader: collect this group's broadcast handles."""
        assert (
            self.parallel_config.data_parallel_rank
            == self.group_leader_dp_rank
        )
        handles: dict[int, Handle] = {}
        for global_dp_rank in self.group_dp_ranks:
            key = (
                f"shared_edge/{self.edge_group_id}/broadcast/"
                f"{global_dp_rank}"
            )
            handles[global_dp_rank] = pickle.loads(
                self._dp_group_store.get(key))
        return handles

    def _publish_per_dp_rank_handles(
        self,
        response_handles: list[Handle],
        peer_response_handles: list[list[Handle]],
    ) -> None:
        """Group leader: publish this group's response handles."""
        assert (
            self.parallel_config.data_parallel_rank
            == self.group_leader_dp_rank
        )
        assert len(response_handles) == len(self.group_dp_ranks)
        assert len(peer_response_handles) == len(self.group_dp_ranks)
        for global_dp_rank, h in zip(
            self.group_dp_ranks, response_handles, strict=True
        ):
            key = (
                f"shared_edge/{self.edge_group_id}/response/"
                f"{global_dp_rank}"
            )
            self._dp_group_store.set(key, pickle.dumps(h))
        for global_dp_rank, hs in zip(
            self.group_dp_ranks, peer_response_handles, strict=True
        ):
            key = (
                f"shared_edge/{self.edge_group_id}/peer_response/"
                f"{global_dp_rank}"
            )
            self._dp_group_store.set(key, pickle.dumps(hs))

    # ------------------------------------ shared edge WorkerProc group leader
    def _spawn_shared_edge_worker_if_master(
        self,
    ) -> tuple[dict[str, Any] | None, list[UnreadyWorkerProcHandle]]:
        """On the group leader, spawn the shared edge ``WorkerProc``, wait
        for it to be ready, and publish the per-DP response /
        peer-response handles to the gloo store. Other group members:
        return ``(None, [])`` (they will read the handles from
        the store directly in :meth:`_attach_response_mqs`).

        Returns a ``(ready_dict, unready_workers)`` tuple. The
        ``unready_workers`` list contains the
        ``UnreadyWorkerProcHandle`` of the spawned worker so the
        caller can pass it into the failure cleanup in
        :meth:`_init_executor` (mirroring the upstream
        ``MultiprocExecutor`` pattern)."""
        if (
            self.parallel_config.data_parallel_rank
            != self.group_leader_dp_rank
        ):
            return None, []
        broadcast_handles = self._collect_broadcast_handles()
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        proc_handle = SharedModelWorkerProc.make_worker_process(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=self.edge_group_id,
            distributed_init_method=get_distributed_init_method(
                get_loopback_ip(), get_open_port()),
            input_shm_handle=None,
            shared_worker_lock=shared_worker_lock,
            is_driver_worker=True,
            shared_broadcast_handles=broadcast_handles,
            global_dp_ranks=self.group_dp_ranks,
        )
        unready_workers = [proc_handle]
        ready_dict = SharedModelWorkerProc.wait_for_ready(unready_workers)
        # Keep the proc handle alive so the worker can be shut
        # down via the death pipe on executor shutdown.
        self._worker_proc = proc_handle.proc
        self._worker_death_writer = proc_handle.death_writer
        self._publish_per_dp_rank_handles(
            ready_dict["response_handles"],
            ready_dict["peer_response_handles"])
        return ready_dict, unready_workers

    def _attach_response_mqs(
        self, ready_dict: dict[str, Any] | None
    ) -> None:
        """Attach this executor's own response MQ (reader) and the
        c cloud workers' response MQs (readers). For the group leader
        the handles come from the ``ready_dict`` already in hand;
        for non-master executors the handles come from the gloo
        store."""
        my_dp_rank = self.parallel_config.data_parallel_rank
        if ready_dict is not None:
            own_handle = ready_dict["response_handles"][
                self.group_local_dp_rank]
            peer_handles = ready_dict["peer_response_handles"][
                self.group_local_dp_rank]
        else:
            assert self._dp_group_store is not None
            own_handle = pickle.loads(self._dp_group_store.get(
                f"shared_edge/{self.edge_group_id}/response/{my_dp_rank}"))
            peer_handles = pickle.loads(self._dp_group_store.get(
                f"shared_edge/{self.edge_group_id}/peer_response/{my_dp_rank}"))

        self.response_mqs = []
        if own_handle is not None and len(
                own_handle.local_reader_ranks) > 0:
            self.response_mqs.append(
                MessageQueue.create_from_handle(own_handle, 0))
        for h in peer_handles:
            if h is not None and h.remote_subscribe_addr is not None:
                self.response_mqs.append(
                    MessageQueue.create_from_handle(h, -1))

    # ----------------------------------------------------------- lifecycle
    def shutdown(self) -> None:
        if not getattr(self, "shutting_down", False):
            logger.debug("Triggering shutdown of shared edge workers")
            self.shutting_down = True
            if proc := getattr(self, "_worker_proc", None):
                if death_writer := getattr(self, "_worker_death_writer",
                                            None):
                    death_writer.close()
                self._ensure_worker_termination([proc])
        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []
        if pg := getattr(self, "_dp_group_pg", None):
            try:
                stateless_destroy_torch_distributed_process_group(pg)
            except Exception:
                logger.warning("Error destroying shared edge gloo group",
                               exc_info=True)
            self._dp_group_pg = None

    def _get_output_rank(self) -> int:
        # The shared edge WorkerProc is the only output rank on
        # the edge side.
        return 0

    def register_failure_callback(self, callback: FailureCallback) -> None:
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback
