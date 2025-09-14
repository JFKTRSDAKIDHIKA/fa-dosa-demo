Experiments
===========

Baseline A1 (Mapping Only)
-------------------------

The ``baselineA_A1`` runner optimizes only mapping and fusion while keeping hardware parameters fixed. Minimal hardware bound checks are disabled (``APPLY_MIN_HW_BOUNDS=False``), so the hardware configuration remains constant throughout the search.

