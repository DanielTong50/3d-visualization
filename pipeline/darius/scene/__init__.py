"""The scene package: the one contract between the compiler and every consumer.

A scene package is a folder holding `scene.json` (validated by `schema.Scene`) and
the asset files it names. The browser simulator, the robotics exporters and the
MuJoCo environment all read packages; only `darius.pipeline` writes them.
"""
