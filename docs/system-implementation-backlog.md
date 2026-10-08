# Implementation roadmap — public summary

This roadmap summarizes the progression from the local demonstrator to an accepted shared deployment. The detailed implementation backlog, estimates, assignments and task dependencies are retained privately.

| Phase | Purpose | Expected result |
| --- | --- | --- |
| P0: Scope and architecture | Agree the scope and record actual platform capabilities | Clear requirements, supported versions and acceptance criteria |
| P1: Trusted foundation | Establish identity and operating controls | Verified case authorization, workload identity, secrets and permitted network paths |
| P2: Shared core and retrieval | Provide shared data and recoverable processing | Traceable source revisions, durable jobs and permitted-source retrieval |
| P3: AI services and CPE | Govern model execution and project isolation | Verified private serving, model release and Controlled Project Environment activation |
| P4: Acceptance and operations | Verify workflows and operating readiness | Workflow acceptance, recovery, monitoring and reproducible handover |
| P5: Approved extensions | Evaluate optional hosting extensions | Separate approval and acceptance for each secondary or onward service |

The local implementation is a demonstrator. Shared deployment, trusted enterprise identity, complete project isolation and OpenShift AI require their own acceptance evidence. The extension path is Primary AI Lab → Secondary GPU Data Center → Sovereign Cloud, with independent approval at each hop.

See the [architecture and observed inventory](enterprise-implementation-architecture.md) and [deployment guide](../deploy/system/README.md) for the public technical baseline.
