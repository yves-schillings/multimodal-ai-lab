# Fresh local clone verification — 7 October 2026

The unpublished working snapshot based on `6402428` was copied into a local Git snapshot, cloned to a separate directory, installed into a fresh Python virtual environment, and exercised with an independent Compose project and initially empty volumes. The local-only snapshot identifier `b51f0aedb0308378aa8c9863d81d95534b42215e` is a verification identifier, not a downloadable GitHub release.

| Check | Result | Evidence |
| --- | --- | --- |
| Fresh environment installation and unit tests | 63 passed; 15 subtests passed; 108.49 seconds including local cloning and dependency installation | [JUnit results](test-results.xml) |
| Browser core | Passed | [Output](browser-tests.txt) |
| Fresh-volume Compose before restart | 27 / 27 | [Report](compose-before.json) |
| Same Compose project after application restart | 8 / 8 | [Report](compose-after.json) |
| Arbitrary non-root UID with network disabled | Writable store, model imports, image OCR and scanned-PDF OCR passed | [Output](container-portability.txt) |
| Model preparation, image build, Compose, restart and portability run | 157.47 seconds | [Timed summary](reproduction-summary.json) |

Prepared speech weights were acquired with `scripts/prepare_speech.py`. The app used port 8784 and MLflow port 5004 through a verification-only port override; all service paths, application sources and checks otherwise used the cloned snapshot. The regular lab's volumes and existing dossier were not reused or changed.

These are **same-workstation, warm-cache results**. They do not establish cold-cache startup time, reproduction on another machine, a hosted CI result, trusted identity, a provisioned Controlled Project Environment, or OpenShift/OpenShift AI deployment. The Compose demo uses synthetic identities and synthetic data. The original earlier evidence remains unchanged.

The [source file hashes](source-files.json) identify the tested snapshot. Its runtime source hashes are checked against the working source at export. The [manifest](manifest.json) records the public evidence payload hashes. The host name has been removed from JUnit metadata; no test results have been changed.

The [exact tested runtime hashes](runtime-source-files.json) also identify the cloned files. Git normalised line endings in 5 files; normalised content matches the working runtime source in all 50 files.
