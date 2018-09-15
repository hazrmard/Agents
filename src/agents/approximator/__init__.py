"""
Defines the `Approximator` class and sub-classes. `Approximator` uses various
models to learn input -> output mappings.

[[x11, x12, x13,..]         [[y11, y12,..]
 [x21, x22, x23,..]   ==>    [y21, y22..]
 ...]                        ...]

    `indim=3`                   `outdim=2`

Available methods are:

* `Polynomial` function approximation,
* Artificial `Neural` network approximation,
* `Tabular` approximation.

All `Approximator`s have the following API:

* Methods:

  * `update(x, y)`: Takes a 2D array of training features in each row and a 2D
  array of target values.

  * `predict(x)`: Takes a 2D array of features in each row and returns a 2D array
  of predicted values.

* Attributes

  * `weights`: The weights used by that particular. Of different forms for each
  subclass.
  * `indim (int)`: The dimensions of input features.
  * `outdim (int)`: The dimensions of output.
"""

from .approximator import Approximator
from .polynomial import Polynomial
from .neural import Neural
from .tabular import Tabular
