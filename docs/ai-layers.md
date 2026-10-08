# Eight-layer responsibility model

This eight-layer working model (design by Yves Schillings) is applied to Multimodal AI Lab.
It is an architectural decomposition for explaining responsibilities,
not an external standard or a claim that every layer is fully implemented.
The design credit is an attribution, not a statement of legal ownership or an external standard endorsement.
The numbering stays fixed across the deck, article and repository.

| Layer | Responsibility | Components and technology | Current evidence / extension |
| --- | --- | --- | --- |
| **L1 Light Frontend** | Simple experimentation and review interface | C01 browser workspace, HTML/JavaScript | Implemented local interface and a separate browser-only synthetic demonstration; no hosted demonstration URL is asserted |
| **L2 Industrial Frontend** | Shared user interface integrated with enterprise identity and services | C01 production interface variant | Target; the local browser interface does not establish enterprise identity |
| **L3 Central API Layer** | Case access, bounded intake, job submission and service contracts | C02 FastAPI, C04 deployment planning policy, C05 inference gateway | Case API and planning rules Implemented; remote gateway Target |
| **L4 AI Backend** | Speech, extraction, classification, retrieval and generation | C03 job runner; M01 faster-whisper; M02 pypdf; M03 Tesseract; M04 sklearn; M05 retrieval; M06 LLM; M07 assistant | Local adapters and extractive workflow Implemented; shared model serving and generative retrieval Target |
| **L5 Data Foundation** | Original sources, revisions, metadata, search indexes and model artifacts | D01 SQLite case store; D02 files; D03 vector index; D04 artifacts | Local storage Implemented; shared case PostgreSQL/vector/object storage Target |
| **L6 Hybrid IT** | Place workloads in approved desktop, site and remote resource pools | H01 desktop, H02 Docker Compose, H03 private infrastructure, H04 OpenShift AI, H05 secondary GPU services, H06 Sovereign Cloud | Desktop and Compose Implemented; remaining hosting placements Target |
| **L7 Security and Compliance** | Identity, case authorization, human authority, audit and project isolation | A01 simulated identity, A02 access rules, A03 review/approval, A04 audit, A05 Controlled Project Environment | Local simulated access/review controls Implemented; trusted identity and infrastructure isolation Target |
| **L8 MLOps and Lifecycle** | Train, evaluate, register, promote, monitor and restore model versions | M08 MLflow; M04 training pipeline; D04 artifacts | Measured synthetic classifier releases Implemented; enterprise operations and broader evaluation Target |

Security and lifecycle controls apply across the other layers. The diagram's vertical ordering is
a responsibility map, not a requirement that every request traverse all eight layers sequentially.

Read [the component register](components.md), [architecture](architecture.md) and
[deployment evidence](deployment.md) for the exact scope of each current implementation.

The [7 October 2026 closing evidence](evidence/2026-10-07-closing/README.md)
includes bounded image and scanned-PDF OCR, local model publication and checksum-verified
promotion/rollback. This evidence does not establish shared identity, a provisioned CPE,
vector retrieval or an OpenShift AI deployment.
