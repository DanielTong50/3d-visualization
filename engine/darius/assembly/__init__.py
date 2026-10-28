"""The assembly graph: what a manual means, as data.

schema     AssemblyPlan (parts, hardware, joints, steps), observations and beliefs
graph      validation, subassemblies after k steps, comparison of two plans
tracking   a hidden Markov model over manual steps: live filtering and video decoding
guidance   the next move: parts, hardware, orientation, checks
"""
