"""Featurisation: Morgan/ECFP fingerprints plus an RDKit descriptor block.

Geometry (radius, bit width, chirality, descriptor list) is configured and honoured; the fingerprint
family is fixed. Every stage that compares or predicts must featurise identically, so the parameters are
threaded from config rather than defaulted per call site.
"""
