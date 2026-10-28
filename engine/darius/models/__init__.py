"""Clients for the foundation models the compiler calls over the network.

Model output is treated as untrusted input: every call is constrained by a JSON
schema and validated before anything downstream sees it.
"""
