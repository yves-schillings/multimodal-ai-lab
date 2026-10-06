# Deployment stages

## Running today

The public synthetic sandbox is available at https://secloudis-case-ai-lab.subllings.chatgpt.site. It executes in the browser and has no operational case connection, file intake, cloud inference or MLflow server. Its simulated personas are teaching aids.

The Python application runs locally at http://127.0.0.1:8770. Use `Start.cmd` on Windows. Runtime databases, model weights, audio and logs are ignored by Git. The Dockerfile is an unverified development build recipe: its loopback binding and simulated identities intentionally do not expose a shared case service. Docker engine and Kubernetes deployment were not verified on the development machine.

## Primary AI Lab target

Red Hat OpenShift AI is the shared deployment target. Before activating application routes, replace simulated identity with trusted OpenID Connect, implement project/case authorization at every service boundary, provision controlled storage and model-serving identities, and validate supported platform/model versions. Deploy model services and pipelines with resource limits, health checks, controlled egress, artifact provenance and measured release gates.

The current local application is not suitable for exposure through an OpenShift Route. Merely changing the bind address does not establish authentication or isolation.

## Future resource pools

Use only the neutral names **Primary AI Lab**, **Secondary GPU Data Center** and **Cloud Souverain**. A policy-approved private interconnect or VPN can extend capacity. Cloud bursting selects a compatible pool only after classification, permitted processing geography, destination approval, retention, CPE requirements and consumption quotas pass. Each onward transfer is independently governed.

Case/source classification is distinct from geography. No actual hosting region, GPU capacity, sovereign certification or commercial rate is assumed. Unavailable eligible capacity queues or rejects work. Access tokens and inference consumption tokens serve different purposes.

## Future CPE lifecycle

A Controlled Project Environment is activated only for cases requiring enhanced isolation. Establish the owner, case policy, approvals and prerequisites; provision namespace, identities, storage and network policy; configure services and quotas; verify permitted and forbidden routes; activate and monitor. Approved shared speech processing can operate without a dedicated CPE while retaining confidentiality controls.

## Later Azure preparation

Use synthetic data for an initial Azure variant. Real source data and derived transcripts require explicit authorisation for the chosen service, region, identity, access, retention and transfer path. No Azure resources have been provisioned by this release.
