"""Importing this package registers every medchem pipeline stage as a side effect.

The CLI and tests do ``import medchem.stages`` before running a pipeline so the
stage registry is populated. Each stage lives in its topic subpackage and registers under the generic
``discovery`` pipeline -- there is no per-target graph, which is what makes retargeting a config change.
"""

from medchem.data import curate as _curate  # noqa: F401  (registers curate)
from medchem.data import pull as _pull  # noqa: F401  (registers data_pull)
from medchem.eval import harness as _harness  # noqa: F401  (registers evaluate)
from medchem.features import featurize as _featurize  # noqa: F401  (registers featurize)
from medchem.generative import stage as _generative  # noqa: F401  (registers generative)
from medchem.models import qsar as _qsar  # noqa: F401  (registers qsar)
from medchem.models import selectivity as _selectivity  # noqa: F401  (registers selectivity)
from medchem.pipeline import _demo  # noqa: F401  (registers the demo pipeline)
from medchem.structure import receptor  # noqa: F401, E402
from medchem.vls import stage as _vls  # noqa: F401  (registers jak1 vls; optional/composable)
