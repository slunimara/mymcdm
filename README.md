This package is the result of the practical part of [my bachelor thesis](https://stag.upol.cz/StagPortletsJSR168/CleanUrl?urlid=prohlizeni-prace-search&studentSearchOsCislo=R20149&praceSearchTyp=bakalářská) on the [Department of Computer Science, Palacky University, Olomouc](https://www.inf.upol.cz). This package is intended to serve as a tool for handling Multiple Attribute Decision Making problems. It can retrieve and process inputs, normalize alternatives, determine relative weights of criteria and process the inputs using decision methods to determine the decision objective of the model. 

## Use examples

The tool can be used both as a Python package using the API and via the console. The use of the console is more focused on smaller problems, while the use of the API package is focused on all types of problems.

The console portion of the package allows the user to load a matrix of variants and criteria weights, normalize the variants, and finally determine the model objective. The user can go through the entire process of working with the VAV model in the command line, except for the ability to specify criteria weights. These must be specified in advance. This section is more geared towards working with smaller problems as it does not offer as much control when working with the data. The data can only be saved, dumped and loaded. 

The package provides two ways of working with VAV problems. The first way is to create a procedure from the implemented methods. The user imports all the required methods into his module and then can program the whole process himself. The decision process can be adapted to different needs and the user has more control over the results of the functions. 

**Using auxiliary method**
```Python
  import numpy as np
  import mymcdm
  
  types = [True, False, True, False]
  weights = np.array((0.20, 0.15, 0.40, 0.25))
  alternatives = np.array(
      [
          (30, 20, 10, 20),
          (25, 20, 15, 30),
          (25, 25, 5, 10),
          (10, 30, 20, 30),
          (30, 10, 30, 10),
      ]
  )
  
  result = mymcdm.decision(
        alternatives, 
        weights, 
        types, 
        "VECTOR", 
        "TOPSIS")
```

The second method is easier for users with less knowledge of VAV theory. The user can use a helper method called **decision**, which can be found in the **[main.py](mymcdm/main.py)** module. The method has several mandatory and optional parameters. The mandatory parameters are: the variance matrix, the weight vector, the code name of the normalization and decision methods. The optional parameters are: a vector of criteria types, a storage location, and a flag whether to store the result and process information. The only thing this method lacks is the ability to specify weights. I have chosen not to include this in the decision method and require that the criteria weights be predetermined.

**Using API**
```Python
  import numpy as np

  from mymcdm.normalization import vector
  from mymcdm.utils import framing
  from mymcdm.methods import topsis
  
  types = [True, False, True, False]
  weights = np.array((0.20, 0.15, 0.40, 0.25))
  alternatives = np.array(
      [
          (30, 20, 10, 20),
          (25, 20, 15, 30),
          (25, 25, 5, 10),
          (10, 30, 20, 30),
          (30, 10, 30, 10),
      ]
  )
  
  alternatives, types = vector(alternatives, types)
  
  weights = framing.frame_criterions(weights, c_types=types)
  alternatives = framing.frame_alternatives(alternatives, a_types=types)
  
  result = topsis(alternatives, weights, types)
  
  decision_matrix = framing.make_decision_matrix(alternatives, weights)
```
 
## Methods list and codenames
For completeness, I have provided a list of all methods. The auxiliary method as an argument takes the code name of normalization and decision methods.

### Decision making methods
| Code name  | Method name  | References  |
|-------------|--------------|-------------|
| **WSM** | Weighted Sum Model | [4] |
| **WPM** | Weighted Product Model | [4] |
| **TOPSIS** | Technique for Order of Preference by Similarity to Ideal Solution | [4] [5] |
| **VIKOR** | VIKOR | [5] |
| **ELECTRE** | Elimination and Choice Translating Reality | [4] [5] [6] |
| **AHP** | Analytic hierarchy process | [4] [5] |

### Normalization methods
| Code name  | Method name  | References  |
|-------------|--------------|-------------|
| **LOG** | Logarithmic normalization | [1] |
| **MAX** | Max normalization | [1] |
| **LINEAR** | Linear normalization | [2] |
| **MAXMIN** | Max-min normalization | [1] |
| **SUM** | Sum normalization | [1] |
| **VECTOR** | Vector normalization | [1] |

### Weighting methods
| Name  | Method name  | References  |
|-------------|--------------|-------------|
| **ENTROPY** | Entropy weights | [3] |
| **MEAN** | Mean Weight | [3] |
| **PAIRWISE** | Pairwise comparison | [3] [7] [8] |
| **POINT** | Point allocation | [3] |
| **SD** | Standard Deviation | [3] |
| **SVP** | Statistical Variance Procedure | [3] |
| **CRITIC** | Criteria importance through inter-criteria | [3] |

## Input examples

Therefore, for better work with data, I implemented a module containing functions for reading and writing data. The function **load_data** allows to read and process the required input data according to the type. If the key "format" is found in the file, which is equal to the value of "matrix" the loading of variants, weights and types of criteria will take place. If the value of the key "format" is equal to "pairwise", the comparison matrices are loaded and processed using the eigenvector method. The user can then retrieve the required data and make a decision based on it. The result of the analysis can be saved using the **save_result** function (or in the decision method when the "save" and "folder" arguments are specified), which receives an object of the **Result** class and saves it. Then the result of the analysis can be retrieved again using the **load_data** function if the path to the saved result is specified. I also use these methods for the console part of the package. All loaded and saved file formats are of type JSON.

**Example input of decision matrix**
```json
{
  "format": "matrix",
  "types": [true, true, false, true],
  "weights": [0.20, 0.15, 0.40, 0.25],
  "alternatives": [
    [30, 20, 10, 20], 
    [25, 20, 15, 30], 
    [25, 25, 5, 10], 
    [10, 30, 20, 30],
    [30, 10, 30, 10]]
}
```

**[\*] Example input of comparison matrices**
```json
{
  "format": "pairwise",
  "types": [true, true, false, true],
  "criteria": [
    ["1", "4", "3", "7"], 
    ["1/4", "1", "1/3", "3"], 
    ["1/3", "3", "1", "5"], 
    ["1/7", "1/3", "1/5", "1"]
  ],
  "alternatives": [
    [
      ["1", "1/4", "4"], 
      ["4", "1", "9"], 
      ["1/4", "1/9", "1"]
    ],
    [
      ["1", "3", "1/5"], 
      ["1/3", "1", "1/7"], 
      ["5", "7", "1"]
    ],
    [
      ["1", "5", "9"], 
      ["1/5", "1", "4"], 
      ["1/9", "1/4", "1"]
    ],
    [
      ["1", "1/3", "5"], 
      ["3", "1", "9"], 
      ["1/5", "1/9", "1"]
    ]
  ]
}
```

---

## References
[1] Vafaei, N., Ribeiro, R. A., & Camarinha-Matos, L. M. (2016b). Normalization Techniques for Multi-Criteria Decision Making: Analytical Hierarchy Process Case Study. Technological Innovation for Cyber-Physical Systems, 261–269. https://doi.org/10.1007/978-3-319-31165-4_26

[2] Natalja, K., Aleksandras, K., & Kazimieras, Z. E. (2018). Statistical Analysis of MCDM Data Normalization Methods Using Monte Carlo Approach. The Case of Ternary Estimates Matrix. Economic Computation and Economic Cybernetics Studies and Research, 52(4/2018), 159–175. https://doi.org/10.24818/18423264/52.4.18.11

[3] Odu, G. (2019). Weighting methods for multi-criteria decision making technique. Journal of Applied Sciences and Environmental Management, 23(8), 1449. https://doi.org/10.4314/jasem.v23i8.7

[4] Triantaphyllou, E. (2000). Multi-criteria Decision Making Methods: A Comparative Study. Applied Optimization. https://doi.org/10.1007/978-1-4757-3157-6

[5] Uzun, B., Ozsahin, I., Agbor, V. O., & Uzun Ozsahin, D. (2021). Theoretical aspects of multi-criteria decision-making (MCDM) methods. Applications of Multi-Criteria Decision-Making Theories in Healthcare and Biomedical Engineering, 3–40. https://doi.org/10.1016/b978-0-12-824086-1.00002-5

[6] Tzeng, G., & Huang, J. (2011). Multiple Attribute Decision Making: Methods and Applications. CRC Press.

[7] Saaty, T. L., & Hu, G. (1998). Ranking by Eigenvector versus other methods in the Analytic Hierarchy Process. Applied Mathematics Letters, 11(4), 121–125. https://doi.org/10.1016/s0893-9659(98)00068-8

[8] Hwang, C., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications : a State-of-the-art Survey. Springer Verlag.

[\*] Wikipedia contributors. (2021, March 19). Analytic hierarchy process – leader example. Wikipedia. https://en.wikipedia.org/wiki/Analytic_hierarchy_process_%E2%80%93_leader_example#cite_note-SYNTHESIZING-9

---