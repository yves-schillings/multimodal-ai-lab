# OKD CPE configuration and activation

A Controlled Project Environment (CPE) is an operator-approved project boundary whose application stays inactive until its controls pass verification. Two local profiles use fictional documents, isolated namespaces and the actual FastAPI application. The baseline approves document extraction, human review and lexical retrieval. The optional AI profile adds **offline BGE-M3 embeddings and Qwen3:4b cited generation** through a controlled inference gateway. Both profiles deny speech, classifier training and model release; those remain in the main lab.

## How the Python code configures OKD

The operator runs Python scripts that construct Kubernetes objects as JSON and submit them with `oc apply`. OKD applies the resulting namespace, identity, network, resource, Secret and storage configuration. Python application modules enforce sessions, CPE state and approved model access. Separate acceptance scripts verify these controls.

These Kubernetes control families are a common foundation for Red Hat OpenShift. The demonstrator controller explicitly permits only `api.crc.testing`. Reusing the design on another OpenShift cluster requires adapting the controller, its operational configuration and its acceptance checks.

## Operator prerequisites

Use the existing local CRC administrator kubeconfig, never a credential copied into the application. The controller permits only `api.crc.testing`. Keep generated passwords and evidence in an owner-only private directory **outside the Git checkout**. Default developer credentials cannot provision namespaces. Do not print kubeconfig contents, passwords or Secret data.

## Provision, verify and activate

Build the runtime sources with `scripts/build_local_app.py` only after source changes. Existing accepted images do not need another build. The controller reads the local ImageStream and pins its immutable digest.

```powershell
python scripts/local_cpe.py provision --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir> --approval-reference <synthetic-owner-reference>
python scripts/local_cpe.py provision --project b --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir> --approval-reference <synthetic-owner-reference>
python scripts/cluster_cpe_acceptance.py --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/cluster_cpe_lifecycle_acceptance.py --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/cluster_cpe_acceptance.py --project b --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/cluster_cpe_lifecycle_acceptance.py --project b --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
```

Both projects must exist before cross-project verification. Verification admits only the reviewer while tenants remain inactive. The lifecycle exercise activates from fresh passing evidence, checks denial and persistence, then leaves the passing project active. Run it without another workflow in that project. A failure revokes the project and preserves storage/evidence.

For normal activation after a fresh passing verification:

```powershell
python scripts/local_cpe.py activate --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
```

Activation requires all checks passing, a matching profile and protected-resource fingerprint, verification less than ten minutes old and unexpired approval. It records the evidence file's SHA-256 in the read-only operator control. Credentials, approval and audit evidence stay private. Approval expires after 24 hours; this is a local exercise, not unattended permanent hosting.

## Actual boundaries and limits

| Control | OKD demonstrator implementation |
| --- | --- |
| Identity | Separate randomly generated project passwords, scrypt hashes in project Secret, persistent server sessions. Runtime service account token not mounted. |
| Network | Default-deny ingress and egress. Same-namespace application access on TCP 8770. The AI profile additionally permits only the scoped gateway, vector database and cluster DNS. No Route, Internet, other project's application or direct Ollama access. Port-forward is an operator path that bypasses NetworkPolicy. |
| Resources | Container requests 500 mCPU / 512 MiB, limits 1 CPU / 1 GiB. Namespace admission budgets constrain requests, limits, GPU allocation, PVC count and storage request. |
| Secrets | Read-only credentials mount. Runtime account cannot list Secret objects or change operator configuration. No Kubernetes credentials in the application. |
| Storage | A distinct 1 GiB PVC request per project. SQLite, originals and audit records persist across pod replacement. CRC hostpath shares physical disk capacity; this is **admission enforcement, not a physical 1 GiB filesystem quota**. |
| Models | Baseline: no approved endpoints. AI profile: exactly two pinned pretrained models, authenticated gateway, bounded requests and one shared inference slot. Audio, training, release, model downloads and arbitrary remote models remain denied. |
| Workload contract | Restricted pod security, immutable approved image, fixed service account, no mounted API token, no host network/PID/IPC, no init/ephemeral containers, allowed volume sources and read-only root filesystem. The temporary volume is operator-managed at 64 MiB and included in the verification fingerprint. |

## Source code by control family

Each link has a corresponding French explanation. The workload contract spans the six families. Verification scripts are listed separately from configuration and runtime enforcement.

| Control | Python source code | Explication en français |
| --- | --- | --- |
| 1. Identities and access rights | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/local_cpe.py">Project account provisioning</a> — <code>scripts/local_cpe.py</code> (<code>Controller.provision</code>)</li></ul> | Crée les comptes propres à chaque projet, génère leurs mots de passe et place les empreintes dans un Secret. Crée le ServiceAccount du traitement et désactive le montage automatique de son jeton Kubernetes. |
| 1. Identities and access rights | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/lab/auth.py">Password hashing and sessions</a> — <code>lab/auth.py</code> (<code>password_hash / LocalSessions</code>)</li></ul> | Calcule les empreintes scrypt, vérifie la connexion et conserve les sessions dans SQLite. Ce module identifie l’utilisateur. Les droits sur les dossiers restent vérifiés séparément par l’application. |
| 2. Network policies | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/local_cpe.py">Baseline network policies</a> — <code>scripts/local_cpe.py</code> (<code>Controller.provision</code>)</li></ul> | Crée les NetworkPolicy qui refusent les communications par défaut et autorisent le chemin applicatif prévu dans le namespace. Le plugin réseau du cluster applique ces règles. |
| 2. Network policies | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/extend_cpe_rag.py">AI profile network paths</a> — <code>scripts/extend_cpe_rag.py</code> (<code>extend</code>)</li></ul> | Ajoute les chemins réseau vers la passerelle d’inférence, la base vectorielle et le DNS du cluster. Configure aussi les règles des services destinataires. Le traitement ne reçoit pas de route directe vers Ollama. |
| 3. Resource and consumption quotas | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/local_cpe.py">Namespace and container budgets</a> — <code>scripts/local_cpe.py</code> (<code>Controller.provision</code>)</li></ul> | Crée ResourceQuota et LimitRange, puis définit les réservations et limites CPU et mémoire des conteneurs. Limite aussi les demandes de stockage, le nombre de volumes et l’allocation GPU. |
| 3. Resource and consumption quotas | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/lab/inference_gateway.py">Inference request limits</a> — <code>lab/inference_gateway.py</code> (<code>create_gateway</code>)</li></ul> | Limite chaque projet à 32 requêtes d’inférence par minute, borne les entrées et paramètres de sortie et utilise un seul créneau d’inférence partagé. Ce mécanisme n’est pas un quota comptable de tokens. |
| 4. Project secrets | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/local_cpe.py">Identity Secret and mounts</a> — <code>scripts/local_cpe.py</code> (<code>Controller.provision</code>)</li></ul> | Crée le Secret des comptes et le monte en lecture seule. Le conteneur reçoit les fichiers nécessaires à son fonctionnement, sans fichier de configuration ni jeton d’administration Kubernetes. |
| 4. Project secrets | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/extend_cpe_rag.py">Database and gateway credentials</a> — <code>scripts/extend_cpe_rag.py</code> (<code>extend</code>)</li></ul> | Crée des identifiants distincts pour la base vectorielle et la passerelle de chaque projet, puis monte les Secrets clients en lecture seule. La rotation nécessite une opération coordonnée et une nouvelle vérification. |
| 5. Controlled storage | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/local_cpe.py">Project persistent volume</a> — <code>scripts/local_cpe.py</code> (<code>Controller.provision</code>)</li></ul> | Crée une demande de volume persistant de 1 GiB par projet et monte les données de l’application. Cette demande limite l’admission du stockage, pas la consommation physique sur le disque partagé. |
| 5. Controlled storage | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/extend_cpe_rag.py">Separate vector databases</a> — <code>scripts/extend_cpe_rag.py</code> (<code>extend</code>)</li></ul> | Crée une base PostgreSQL et un rôle non administrateur pour chaque projet. Retire les droits de connexion publics afin d’empêcher l’accès aux bases des autres projets. |
| 6. Approved model endpoints | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/extend_cpe_rag.py">AI profile configuration</a> — <code>scripts/extend_cpe_rag.py</code> (<code>extend</code>)</li></ul> | Déploie la passerelle FastAPI et configure son adresse, les identifiants des projets et le profil RAG. Révoque les projets avant de modifier leurs contrôles protégés. Cette étape ne les active pas. |
| 6. Approved model endpoints | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/lab/inference_gateway.py">Model and operation checks</a> — <code>lab/inference_gateway.py</code> (<code>create_gateway</code>)</li></ul> | Vérifie le jeton du projet, l’opération demandée et l’empreinte des modèles approuvés. BGE-M3 produit les vecteurs de recherche et Qwen3:4b génère les réponses. Les téléchargements et le repli vers un service distant sont refusés. |
| 6. Approved model endpoints | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/lab/cpe.py">Application activation gate</a> — <code>lab/cpe.py</code> (<code>CpeGate</code>)</li></ul> | Relit l’état du CPE et la validité de son approbation avant les opérations applicatives. Refuse l’accès si le contrôle est absent, invalide, expiré ou révoqué, et restreint les actions aux modèles du profil approuvé. |
| Cross-cutting workload contract | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/local_cpe.py">Pod restrictions and activation evidence</a> — <code>scripts/local_cpe.py</code> (<code>Controller.admission / snapshot / activate</code>)</li></ul> | Restreint les pods à l’image et au compte approuvés, aux volumes autorisés et à une racine en lecture seule. Refuse les espaces réseau et processus de l’hôte et les conteneurs additionnels interdits. Compare l’empreinte des ressources aux preuves avant activation. |
| Control verification | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/cluster_cpe_acceptance.py">Functional acceptance checks</a> — <code>scripts/cluster_cpe_acceptance.py</code> (<code>verify</code>)</li></ul> | Teste les connexions permises et interdites, les identités, les autorisations Kubernetes, les quotas, le stockage et les réponses de la passerelle. Écrit les preuves utilisées pour décider de l’activation. |
| Lifecycle verification | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/cluster_cpe_lifecycle_acceptance.py">Revocation and persistence checks</a> — <code>scripts/cluster_cpe_lifecycle_acceptance.py</code> (<code>accept</code>)</li></ul> | Vérifie l’activation à partir de preuves valides, la révocation, l’expiration et le refus de preuves incohérentes. Remplace le pod applicatif et vérifie que le volume, le dossier et la session sont conservés. |
| RAG verification | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/scripts/cluster_cpe_rag_acceptance.py">Both-project RAG acceptance</a> — <code>scripts/cluster_cpe_rag_acceptance.py</code> (<code>accept_all</code>)</li></ul> | Coordonne les contrôles des deux projets, la génération avec citations, le remplacement des applications et le redémarrage du service vectoriel. Vérifie les réponses et l’identité des volumes persistants conservés. |

## Authorized access across several projects

The [article design and grant example](https://secloudis.com/ai-lab-multimodal-case-processing/#cpe-workspace-guide-3) explain independent per-user project, case, document and action grants. The grant registry and delegated cross-project retrieval remain unimplemented and unaccepted. Existing authentication and single-store case checks are supporting components, not evidence of cross-project search.

| Reference | Source code or documentation | Explication en français |
| --- | --- | --- |
| Cross-project access design | <ul><li><a href="https://secloudis.com/ai-lab-multimodal-case-processing/#cpe-workspace-guide-3">Architecture and project grants</a> — <code>Article section 3.4.5</code></li></ul> | Décrit les droits indépendants d’un utilisateur sur plusieurs projets, dossiers et actions. Chaque projet filtre les données avant la recherche. Le portail commun ne nécessite pas un environnement dédié par utilisateur. Le registre de droits et la recherche transversale sont documentés, mais ne sont pas implémentés ni acceptés dans le runtime actuel. |
| Existing identity mechanism | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/lab/auth.py">Authenticated users and sessions</a> — <code>lab/auth.py — LocalSessions</code></li></ul> | Identifie l’utilisateur connecté et vérifie sa session. Ce module constitue une brique existante à réutiliser. Il ne fournit pas à lui seul un registre d’autorisations sur plusieurs projets. |
| Existing case authorization | <ul><li><a href="https://github.com/yves-schillings/multimodal-ai-lab/blob/main/lab/store.py">Case ownership and reviewer access</a> — <code>lab/store.py — StatementStore._case</code></li></ul> | Vérifie les droits du propriétaire du dossier ou du reviewer dans le stockage courant. Ce contrôle est existant, mais il n’effectue pas de recherche fédérée entre plusieurs projets et ne remplace pas les droits multi-projets proposés. |

The same application validates the current phase on each business request, queued job authorization and store actor check. Revoked, expired, malformed or missing controls return HTTP 503. Model actions outside the approved profile return HTTP 403. Case writes still check ownership and source revision. This local operator boundary is separate from organizational identity and production assurance.

Mounted ConfigMaps propagate through kubelet rather than a direct application API. The operator updates a scoped pod annotation to prompt refresh and acceptance waits for the actual observed application phase. This follows the [Kubernetes mounted ConfigMap update mechanism](https://kubernetes.io/docs/tasks/configure-pod-container/configure-pod-configmap/). Do not assume a ConfigMap write alone has already revoked a running session.

## Revocation, image changes and recovery

```powershell
python scripts/local_cpe.py revoke --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
python scripts/local_cpe.py upgrade-image --project a --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-dir>
```

Revocation retains every PVC and evidence file. An image upgrade revokes first, pins the new local digest, updates the scoped admission contract and deployment, then returns to provisioned. Fresh verification is mandatory. The controller refuses to reprovision an existing runtime, avoiding accidental credential or data replacement.

The acceptance scripts separately demonstrate pod-to-pod NetworkPolicy denial and operator port-forward access. A probe execution failure never counts as a successful denial. Lifecycle checks retain the same PVC UID, existing case and session after replacing only the project's application pod. Do not reset the cluster or remove namespaces/PVCs as a routine recovery action.

## Extend both projects to offline RAG

First prepare and accept the main lab's offline vector and inference services using [the RAG guide](local-rag.md). The extension reuses their pinned weights; it does not download models during requests. Build only changed runtime sources, then run:

```powershell
python scripts/extend_cpe_rag.py --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-cpe-dir>
python scripts/cluster_cpe_rag_acceptance.py --admin-kubeconfig <local-admin-kubeconfig> --private-dir <private-cpe-dir>
```

The operator revokes both projects before modifying protected controls. Each gets its own PostgreSQL database and non-administrative role (`cpe_vectors_a`/`cpe_vector_a`, or `cpe_vectors_b`/`cpe_vector_b`). Each role is denied connection to the other project, the main vector database and the administration database. The original case PVCs and accounts are retained. Read-only mounted credentials authenticate project-specific access to the shared gateway; no Kubernetes token is mounted.

The gateway permits only tags, embedding and generation operations for the approved model digests. It denies pull/download operations and cloud fallback, limits each project to 32 inference requests per minute, bounds inputs and serializes CPU inference with a five-second capacity wait. Shared inference does not imply shared case data. Reviewed sources, permission checks, current revisions and checked supporting quotes remain authoritative. Citation checks do not prove factual entailment; human review remains required.

On 9 October 2026, both AI profiles passed 39 verification checks, 13 lifecycle checks and six actual Chromium browser checks each. The wrapper also passed both application-replacement RAG checks and all three vector-restart/PVC checks. See the [curated accepted evidence](../evidence/2026-10-09-local-ai-extensions/README.md). The accepted VM uses six vCPUs and 24 GiB RAM; the preceding 16 GiB run stalled during inference and was not accepted.

Fresh verification checks real cited generation, unsupported-question abstention, replacement of stale indexed text after a source correction, cross-project database denial and all original isolation controls. The lifecycle exercise separately checks revocation, expiry and app replacement. Finally, the wrapper replaces only the vector database pod and checks both projects' answers and the original PVC identity. Actual reports remain private. Approval still expires after 24 hours; changing the approved profile requires fresh acceptance.

## Remaining production scope

Organizational OIDC, TLS ingress, administrator separation, retention/expiry automation and enforced physical storage quotas remain production work. [The lightweight AI platform](local-ai-platform.md) installs separate KServe/Jupyter/training components on this CPU-only OKD cluster; it is not the complete Red Hat OpenShift AI product. GPU execution is not installed.
