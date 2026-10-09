"""Install the separate local pgvector service without exposing or resetting data."""
import base64
import json
import secrets
import subprocess
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "multimodal-ai-lab"


def oc(*args, **kwargs):
    return subprocess.run(["oc", *args], capture_output=True, text=True, timeout=300, **kwargs)


def password(name):
    result = oc("get", "secret", name, "-n", NAMESPACE, "-o", "json")
    if result.returncode == 0:
        return base64.b64decode(json.loads(result.stdout)["data"]["password"]).decode()
    # Check API access separately; do not treat connection/permission errors as missing data.
    oc("get", "secrets", "-n", NAMESPACE, check=True)
    value = secrets.token_urlsafe(36)
    payload = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": name, "namespace": NAMESPACE},
               "type": "Opaque", "stringData": {"password": value}}
    oc("create", "-f", "-", input=json.dumps(payload), check=True)
    return value


def main():
    password("lab-vector-admin")
    client_password = password("lab-vector-client")
    result = oc("apply", "-n", NAMESPACE, "-f", str(ROOT / "deploy/openshift/local-rag/pgvector.yaml"), check=True)
    print(result.stdout.strip())
    result = oc("rollout", "status", "statefulset/lab-vector", "-n", NAMESPACE, "--timeout=240s", check=True)
    print(result.stdout.strip())
    if not re.fullmatch(r"[A-Za-z0-9_-]{40,80}", client_password):
        raise RuntimeError("Unexpected client secret format; no SQL executed.")
    # A generated URL-safe secret enters SQL only through stdin, never command arguments/logs.
    sql = """
CREATE EXTENSION IF NOT EXISTS vector;
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='vector_client') THEN
    CREATE ROLE vector_client LOGIN;
  END IF;
END $$;
ALTER ROLE vector_client NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD '%s';
GRANT CONNECT ON DATABASE lab_vectors TO vector_client;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
CREATE SCHEMA IF NOT EXISTS retrieval AUTHORIZATION vector_client;
""" % client_password
    oc("exec", "-i", "-n", NAMESPACE, "lab-vector-0", "--", "psql", "-v", "ON_ERROR_STOP=1",
       "-U", "vector_admin", "-d", "lab_vectors", input=sql, check=True)
    print("pgvector installed; dedicated non-superuser client/schema configured. Retrieval acceptance still required.")


if __name__ == "__main__":
    main()
