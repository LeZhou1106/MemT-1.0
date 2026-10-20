"""Wire-format values that the client and the server must agree on."""

HTTP_PROTOCOL = "memt-http-v1"
SNAPSHOT_FORMAT = "memt-memory-v1"
# Session IDs and RID nonces match [A-Za-z0-9_.-]{1,MAX_IDENTIFIER_LENGTH};
# both sides validate RIDs with protocol/requests.py.
MAX_IDENTIFIER_LENGTH = 160
# Lease "ttl" of /memt/control, in seconds: the server default and upper bound.
DEFAULT_LEASE_SECONDS = 300
MAX_LEASE_SECONDS = 3600
