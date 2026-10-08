# Closing verification — 7 October 2026

This package describes the tested local working snapshot based on commit `6402428`. It is not a published release and does not represent a hosted CI run.

| Check | Evidence | Scope |
| --- | --- | --- |
| Python | [closing-tests.xml](closing-tests.xml) | 63 tests and 15 subtests passed |
| Browser core | [browser-core.txt](browser-core.txt) | Gates, source eligibility and abstention |
| Compose before restart | [closing-before.json](closing-before.json) | 27 checks passed |
| Compose after restart | [closing-after.json](closing-after.json) | 8 checks passed |
| Fictional dossier | [dossier-evidence.json](dossier-evidence.json) | Real CPU speech, scanned PDF, review, citations, denials, invalidation and planning policy |
| Arbitrary UID | [container-portability.txt](container-portability.txt) | Actual image and scanned-PDF OCR, network disabled |
| Python requirements | [closing-pip-audit.json](closing-pip-audit.json) | No known vulnerabilities reported |
| Application image | [app-audit.json](app-audit.json) | 76 HIGH and 1 CRITICAL package findings |
| Source snapshot | [source-snapshot.json](source-snapshot.json) | File hashes of the local code and deployment snapshot |

The scanner was Trivy 0.75.0, executed as image digest `sha256:af6acf9a6b85dfe389a1941505c0ce9efef52a4719635e1a962f022a3d855daa` against the exported application image `17281945386e`. The scanner supplied no fixed versions for these findings. The image security gate therefore fails. Review reachable use and vendor remediation before shared exposure. No findings were suppressed.

Only synthetic content appears in the dossier and screenshots. A synthetic receipt was misclassified with confidence about 0.29 despite perfect tiny held-out scores. The raw record keeps this counterexample.

The tests ran on the development workstation with prepared speech weights and persistent Compose services. This is not a clean-machine or cold-cache reproduction. No representative human French or Dutch benchmark, shared identity acceptance, provisioned CPE, OpenShift cluster or OpenShift AI installation is claimed. Those require separate records.

Architecture figures remain the original issued figures. Any simplified proposal is separate and requires approval. All package payload hashes are in [manifest.json](manifest.json).

## French and Dutch machine-generated fixtures

`speech-fr-nl.json` records the actual offline run, exact reference and recognised text, individual and aggregate WER, model hashes and audio hashes. `speech-fixture-specification.json` is the fictional source corpus and `speech-audio/` contains all twenty tested clips. French WER is 67.11% and Dutch WER 71.01%. The result does not establish representative human-speech quality. See ../../speech-evaluation.md for method and reproduction.
