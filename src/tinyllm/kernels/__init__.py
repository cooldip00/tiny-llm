"""Triton kernels (PLAN 4.10, D-016): each has a PyTorch reference it must match (`allclose`).

Importing a kernel module needs Triton (Linux); callers fall back to PyTorch without it.
"""
