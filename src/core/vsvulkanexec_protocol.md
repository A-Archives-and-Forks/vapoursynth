# GPU exec pool protocol and invariants

Internal design reference for the Vulkan execution pools: `vsvulkanexec.h/.cpp`, the pool
registry, release batches and admission gate in `vsvulkan.h/.cpp`, the reclamation ladder in
`vsvulkanalloc.cpp`, the public wrappers in `vsapi.cpp` and the core's cache sweep in
`vscore.cpp`. It states what the code guarantees, which lock guards what, and the invariants
every change must keep. The public contract it describes is the one in `VSVulkan4.h`; when the
two disagree, the header is what plugins were promised and this file is what needs fixing.

Verified against the tree of 2026-09-05 (the release-batch registry with sequence numbers and
the enforced rules I15 to I21). Section 11 lists invariants that are *not* enforced yet and
what each would rule out.

## 1. Objects and ownership

| Object | Owned by | Holds |
|---|---|---|
| `VSVulkanDevice` | the core, and every frame or pool still holding a reference; may outlive the core | the pool registry, the release-batch registry, the retained-bytes total and budget, the progress timeline, the allocator |
| `VSVulkanExecPool` | a filter through `VSGPUExecPool`, or `VSVulkanTransfer` | a timeline (counted; frames it produced hold their own references), `contextCount` contexts, `nextValue`, its registration |
| `VSVulkanExecContext` | its pool | a command pool with one primary buffer, the `claimed` flag, `pendingValue`, the retention list, `retainedBytes`, `retainedCounted` |
| retention | the context it was registered on, until reaped | a release function and an object; bytes are summed on the context |
| release batch | the device registry | pool, running thread, sequence number |
| `VSGPUExecContext` | the caller, from `gpuExecAcquire` to `gpuExecSubmit`/`gpuExecAbandon` | the claimed context, the deduplicated wait list, the planes to publish |

A public pool's ring has `clamp(workerThreads, 2, 8)` contexts, fixed at creation. The transfer's
pool has one context per staging slot and never retains anything.

## 2. Locks and their order

| Lock | Guards |
|---|---|
| `VSVulkanDevice::execPoolsMutex` + `execReleaseCv` | the pool registry, the batch registry, `nextExecReleaseBatch`, creation of the progress semaphore |
| `VSVulkanExecPool::claimMutex` + `claimCv` | nothing but the rendezvous between a full ring and `releaseClaim`; the claim itself is the atomic `claimed` |
| `VSVulkanQueue` | `vkQueueSubmit2`, `nextValue`, `execProgressNext` |
| allocator mutex | blocks and free lists |
| `VSCore::cacheLock` | the set of nodes with caches |
| `VSNode::cacheMutex` | one node's cache and consumer list |

Order, outermost first: `execPoolsMutex` before `claimMutex` (a device sweep releases claims
while walking the registry) and before the queue lock (`detachCompleted` reads `nextValue` for
the counter check); `cacheLock` before `cacheMutex` (eviction); the queue lock and the allocator
mutex take nothing themselves. `registerCache` is called
after `cacheMutex` is released, never under it. **No lock in this table is held while a release
callback runs**, and `execPoolsMutex` is never held while calling into anything a plugin wrote.

## 3. Context life cycle

```
fresh ──acquire──▶ recording ──submit ok──▶ pending ──GPU done──▶ completed ──reap──▶ idle
   ▲                  │                        (claimed by                  (unclaimed,
   │             abandon / submit               nobody)                      list moved out)
   └───────────────failed──────────────────────────────────────────────────────┘
```

- **acquire** (`VSVulkanExecPool::acquire`): pass the admission gate, then claim a context by
  CAS, first from the ring cursor, then under `claimMutex` when the ring is full. If the
  context has a `pendingValue`, register a batch when its retention list is non-empty, wait the
  pool timeline for that value, release the retentions (`releaseRetainedNow`), end the batch.
  Then reset the command pool and begin the buffer. Every failure path drops the claim.
- **recording**: `retain` appends to the context's list and adds to `retainedBytes`; nothing
  reaches the device yet. Only the claim holder may call it.
- **submit** (`submit`): end the buffer (failure releases retentions and claim); deduplicate the
  waits; under the queue lock allocate the next timeline value and, for compute-queue pools,
  the next progress value, and submit. On success add `retainedBytes` to the device total and
  set `retainedCounted`, still under the claim; on failure release the retentions (uncounted).
  Drop the claim, then reap the pool's other completed contexts (`sweepCompleted`).
- **abandon**: release the retentions (uncounted) and drop the claim.
- **completed**: a context nobody holds whose `pendingValue` the timeline has reached. Any
  reaper may claim it by CAS and detach its retentions.
- **destruction** (`~VSVulkanExecPool`): unregister (fatal from inside a release batch; waits
  for every other thread's batch for the pool), fatal if any context is still claimed,
  `waitAll`, release what the contexts still hold, destroy the command pools, drop the
  timeline reference.

## 4. Reaping and the batch registry

Who runs releases, and from where:

| Reaper | Trigger | Reaches |
|---|---|---|
| device sweep `sweepExecPools` | `notifyCaches` (before `cacheLock`), each admission gate loop, rungs of the allocation ladder | every registered pool's completed, unclaimed contexts |
| pool sweep `sweepCompleted` | end of `submit`, `waitAll` | the pool's completed, unclaimed contexts |
| `acquire` | claiming a context with a pending submission | that context |
| `abandon`, failed `submit` | the caller | that recording's uncounted retentions |
| destructor | `freeGPUExecPool` | everything left |

`detachCompleted` reads the pool timeline once, checks the counter against `nextValue` (read
afterwards, under the queue lock; a counter past it means the semaphore was signalled from
outside and is fatal), skips claimed contexts, wins the claim by CAS on the rest, and for each
completed one settles the bytes and moves the list out before dropping the claim. Releases run
only after that, from a local list, with no lock held.

Every batch of releases is registered on the device for as long as it runs, as `{pool, thread,
id}`, whichever reaper runs it; `acquire` registers from the moment it wins the claim, because
from then on no sweep can reach that context. The waits built on the registry:

| Wait | Waits for | Bound | Own-thread batch for the pool |
|---|---|---|---|
| `unregisterExecPool` | every other thread's batch for the pool | none | fatal |
| `waitExecReleases` (from `waitAll`) | every other thread's batch for the pool | none | fatal |
| `waitForeignExecReleases` (allocation ladder) | other threads' batches of any pool that existed when the wait began | none | excluded |

The two public promises map onto these: `freeGPUExecPool` returns only after every release the
pool registered has run; `gpuExecPoolWaitIdle` waits the timeline, reaps, then waits for
releases other threads got to first.

## 5. Accounting

`execRetainedBytes` is the sum of `retainedBytes` over contexts whose `retainedCounted` is set.
Bytes enter at a successful submit and leave when the retention is detached for release, before
any callback runs. Recordings in progress, abandoned recordings and failed submissions never
reach the total. The budget is a quarter of the VRAM limit, refreshed by `setMaxVRAMUse`; zero
disables the gate.

What each typed retention counts: a GPU-resident frame its whole size, a host frame nothing, a
pooled buffer or region its region size, a user object whatever the caller passed. A pool that
does not signal progress meters nothing (I21): its retentions are kept and released as usual
but never enter the total. The allocator's blocks are accounted separately through
`accountAllocation` into `MemoryUse`; transfer staging buffers are not accounted anywhere.

## 6. Admission gate

`execAdmissionGate`, entered by every `acquire`, returns at once when the total is within
budget. Otherwise it loops: sample the progress counter, sweep the device, return if within
budget, wait for `counter + 1` with a 50 ms timeout, repeat. The counter is sampled *before* the
sweep so a completion between the two either gets reaped or leaves the counter behind the wait
target. Only submitted work is counted, so a thread's own recordings never gate it. Compute-queue
pools signal the progress timeline on every submission; a pool on a dedicated transfer queue
does not, and its retentions are not metered. The timeout remains for the one event that
reduces the total without a signal: a completed context an acquirer claimed first settles its
bytes on the host. A host-signalled wake-up (P10) would retire it.

## 7. Allocation ladder

`allocatePooled` retries after each rung:

1. sweep the device, wait for the releases other threads had in flight, sweep again;
2. the pressure callback, which evicts every cached GPU frame and trims idle blocks;
3. fail.

Frame creation turns a failure into `logFatal`, so a rung that gives up while another thread's
release is a microsecond from freeing the memory is a crash, not a slow frame. That is why
rung 1 waits.

## 8. Public contract (summary of `VSVulkan4.h`)

- Every acquire ends in exactly one submit or abandon, by the thread that acquired; retain,
  submit and abandon from any other thread are fatal.
- One context of a pool per thread, fatal otherwise; contexts of different pools may be held
  together. The in-flight budget counts submitted work only.
- Retain only between acquire and submit.
- A release callback runs on whichever thread reaps it, possibly inside the filter's own
  `getFrame` (a gated acquire or a failed allocation sweeps), with no core lock held. It may take
  its own locks and free; acquiring, allocating GPU memory, and creating, freeing or waiting on
  a pool from inside it are fatal.
- `gpuExecPoolWaitIdle` only from a thread holding no context of the pool, fatal otherwise.
- Pools are destroyed in the filter's free callback, with no context claimed; destroying one
  while holding its own context is fatal.
- Never signal a pool's timeline from outside.

## 9. Invariants

| # | Invariant | Kept by |
|---|---|---|
| I1 | A context is touched by exactly one party at a time: its claim holder, or a reaper that won the CAS. | the atomic claim; sweeps skip claimed contexts |
| I2 | A retention list is appended only by the claim holder and moved out only under the claim. | construction; no code reads a list unclaimed |
| I3 | Every retention's release runs exactly once, after its submission completed or after the recording was abandoned or failed. | lists are moved out before running; `pendingValue <= counter` gates detach |
| I4 | `execRetainedBytes` equals the sum over counted contexts and never underflows. | add under the claim at submit; subtract once at settle under the claim; uncounted paths never subtract |
| I5 | Every batch of releases is registered on the device for its whole duration, from the claim in `acquire`'s case. | begin/end around every `runReleases` site |
| I6 | `freeGPUExecPool` and `gpuExecPoolWaitIdle` return only when no other thread holds a batch for the pool; an own batch is fatal, never silently skipped. | `unregisterExecPool`, `waitExecReleases`, `failIfInsideOwnBatch` |
| I7 | No lock from section 2 is held while a release callback runs. | sweeps detach under the lock and run after it; `notifyCaches` sweeps before `cacheLock`; the ladder and the gate hold nothing |
| I8 | The lock order in section 2. | construction; `registerCache` after `cacheMutex` is released |
| I9 | The gate waits only on bytes that submitted work will release, and never sleeps past a completion it could have reaped. | bytes counted at submit; counter sampled before the sweep |
| I10 | The ladder never declares the device full while a release another thread already detached is still running. | rung 1's wait and second sweep |
| I11 | Timeline values are allocated and submitted under the queue lock, strictly increasing per pool; `pendingValue` is the context's last submitted value or zero. | `submit` |
| I12 | A destroyed pool has no registered batch on any thread, no claimed context and empty lists, and is off the registry before anything is torn down. | destructor order |
| I13 | The transfer pool never retains, so it never registers batches and never contributes bytes. | `VSVulkanTransfer` uses no retention |
| I14 | A retained GPU frame outlives its submission; its bytes are its whole size. | `vkGPUExecReadsFrame` |
| I15 | A release callback only frees: no acquire, GPU allocation, pool creation, pool free or pool wait from a thread running a batch. | `failIfRunningReleases` in `acquire`, `allocatePooled`, `registerExecPool`, `unregisterExecPool`, `waitAll` |
| I16 | One context per pool per thread. | the owner thread recorded on the claim; `failIfHoldingContext` in `acquire` |
| I17 | Retain, submit and abandon only by the claim holder's thread. | `failUnlessOwner` |
| I18 | `waitAll` (idle wait and destruction) only from a thread holding no context of the pool. | `failIfHoldingContext` in `waitAll` |
| I19 | A pool is never destroyed while any thread holds one of its contexts. | `failIfAnyContextHeld` in the destructor, after unregistration, when a claim can only mean a thread still using the pool |
| I20 | The pool's timeline advances only through the pool's own submissions: its counter never exceeds `nextValue`. | the check in `detachCompleted`, counter read first and `nextValue` after it under the queue lock |
| I21 | Every metered byte belongs to a submission whose completion signals the progress timeline. | `retain` adds bytes only on a pool with `signalsProgress` |

## 10. Known windows and their bounds

| # | Window | Consequence | Bound |
|---|---|---|---|
| W1 | Bytes leave the total at detach, the region returns at the callback. | a gate-admitted thread can fail at the driver | rung 1 waits for the release and retries |
| W6 | A completed context claimed by an acquirer settles its bytes on the host, with no signal to the gate. | a gated thread wakes only at the next completion or the poll | 50 ms |

W2, W4 and W5 of the earlier revision are gone with I15 and I18: a batch can no longer wait on
anything, so the ladder's wait is unbounded, and a caller cannot wait a pool idle around its
own recording. W3 is gone with I21. W6 is what the gate's poll now exists for.

## 11. Candidate invariants, not enforced

P1 to P4 of the earlier revision are now I15 to I18, P6 and P9 are I19 and I20, and P5 is I21
(as "not metered" rather than fatal, since the typed calls count bytes on their own and a
transfer pool that copies GPU frames has to read them). Each of the rest would delete a further
class of interleavings from what must be reasoned about, and each is cheap.

| # | Candidate | Enforce by | Eliminates |
|---|---|---|---|
| P7 | Per-pool byte identity. | keep a per-pool counted sum; assert zero at pool destruction and a zero device total at device destruction, in a self-check mode beside `VS_VULKAN_VALIDATION` | accounting drift going unnoticed until the gate stalls |
| P8 | No lock held during callbacks, checked rather than argued. | debug wrappers on `cacheLock` and `execPoolsMutex` that record the owner; `runReleases` asserts the current thread owns neither | regressions of I7 by a future caller |
| P10 | Every event that reduces the byte total wakes the gate. | a second timeline that only the host signals, once per settle; the gate waits on it and the progress timeline with `VK_SEMAPHORE_WAIT_ANY_BIT` | W6, and with it the gate's 50 ms poll, which can then go entirely |

With I15 to I18 in place callbacks are pure frees and every rendezvous is unambiguous, which
is the shape a single-reaper design would enforce structurally; that later change, if wanted,
is now a refactor rather than a redesign.
