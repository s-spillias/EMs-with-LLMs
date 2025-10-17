---
abstract: |
  Marine ecosystem modeling faces increasing challenges in balancing
  model complexity with practical utility for fisheries management,
  particularly as climate change introduces new uncertainties. We
  present AIME (AI for Models of Ecosystems), a novel framework that
  combines evolutionary algorithms with artificial intelligence and
  advanced statistical inference techniques. AIME automates the
  generation and refinement of ecological models through an iterative
  process that maintains biological plausibility while optimizing model
  performance. The framework employs retrieval-augmented generation for
  literature-based parameter initialization, implements multi-phase
  optimization strategies, and an evolutionary mechanism to identify
  numerically and ecologically feasible models. We demonstrate AIME's
  application to Crown-of-Thorns starfish populations on the Great
  Barrier Reef, modeling their interactions with fast and slow-growing
  coral species. This approach represents a significant advance in
  ecosystem modeling by bridging the gap between simple single-species
  models and complex whole-ecosystem approaches, while providing a
  flexible, efficient framework for developing management-relevant
  ecological models in rapidly changing environments.
author:
- Scott Spillias^1,2^[^1]
bibliography:
- references.bib
date: 2025-03-31
title: An AI-Driven Framework for Automated Generation of Marine
  Ecosystem Models
---

^1^CSIRO Environment, Hobart, Australia\
^2^Centre for Marine Socio-Ecology, University of Tasmania, Hobart,
Australia\

**Keywords:** Artificial Intelligence, Ecological Modeling, Evolutionary
Algorithms, Large Language Models, Marine Ecosystems, Crown-of-Thorns
Starfish, Great Barrier Reef, time-series forecasting

# Introduction

Ecosystem models provide essential tools for managing complex
interactions between nature and people in ecological systems
[@McCarthy2004; @Holden2016]. The development of these models
traditionally requires significant time and expertise, creating a
bottleneck in addressing urgent environmental challenges
[@Dichmont2017; @Holden2024b]. This limitation has become particularly
acute as climate change and other anthropogenic pressures demand rapid,
adaptable modeling approaches for ecosystem management
[@weiskopf2020climate; @malhi2020climate].

Artificial Intelligence offers promising solutions to these modeling
challenges, with potential to accelerate model development and enhance
adaptability [@Spillias2024]. While initial efforts to apply AI in
ecological modeling focused on machine learning approaches that rely on
black-box methods [@morales2024developing], emerging techniques in
equation discovery and automated scientific discovery show particular
promise [@Huntingford_Nicoll_Klein_Ahmad_2024; @floryan2022data]. These
methods can derive interpretable mathematical relationships directly
from data, offering advantages over statistical emulators when modeling
novel environmental conditions
[@Schaeffer_2017; @chen2024constructing; @karniadakis2021physics].
Similarly, attempts to leverage large language models for direct
time-series prediction
[@zhang2024large; @su2024large; @hassani2024predictions; @gandhi2024generative; @bylund2024chatgpt; @cao2023tempo; @li2024lite],
though successful in other fields, are unsuited for producing reliable
ecological insights or testing management interventions. For instance,
recent work on multimodal LLMs for environmental prediction
[@li2024lite] achieves impressive accuracy in forecasting physical
variables like streamflow and water temperature, but does not address
the mechanistic relationships needed for ecosystem management.

Rather than using AI to replace traditional modeling approaches, recent
advances in AI coding capabilities suggest a more promising direction
[@Xu2021]. Language models like o3-mini and Claude can assist in
constructing mechanistic models [@Spillias2024], maintaining
interpretability while accelerating the development process
[@TheRoyalSociety2024]. Recent demonstrations of LLMs automating
scientific processes, from autonomous chemical experimentation
[@burger2023autonomous] to biomedical research [@wang2024bioresearcher],
and even fully automated scientific discovery [@kramer2023automated],
highlight their potential for systematic scientific work. The key
challenge lies in developing frameworks that can systematically harness
these capabilities while ensuring scientific rigor and maintaining human
oversight in the discovery process [@kramer2023automated]. To the best
of our knowledge, such an approach has not been attempted yet in
ecological modeling.

To address this challenge, we present \"AI for Models of Ecosystems\"
(AIME), a novel framework that integrates large language models with
evolutionary optimization to automate the discovery of interpretable
ecological models. AIME employs Template Model Builder (TMB) as its
foundation, providing a rigorous statistical framework for ecological
modeling. The system operates through an iterative process where an LLM
generates candidate model structures as TMB-compatible equations, which
are then evaluated against time-series data using normalized objective
functions. These models undergo evolutionary optimization, where
successful structures are selected and combined to produce increasingly
accurate representations of ecosystem dynamics.

We validate AIME through two complementary case studies that test
different aspects of ecological modeling. First, we evaluate the
framework's ability to recover fundamental ecological theory using
synthetic data generated from a well-established
nutrient-phytoplankton-zooplankton (NPZ) model
[@edwards1999zooplankton]. This controlled experiment tests AIME's
equation-learning capabilities by comparing discovered equations against
known mathematical relationships that represent core ecological
processes. Second, we assess AIME's ability to address
management-relevant predictions using synthetic data based on
Crown-of-Thorns starfish (COTS) populations on the Great Barrier Reef,
derived from existing MICE models
[@morello2014model; @Rogers_Plaganyi_2022; @Plaganyi_Punt_Hillary_Morello_Thebaud_Hutton_Pillans_Thorson_Fulton_Smith_et_al_2014; @Condie_Anthony_Babcock_Baird_Beeden_Fletcher_Gorton_Harrison_Hobday_Plaganyi_et_al_2021].
The COTS case study tests the framework's robustness to noisy data while
focusing on a specific management challenge: predicting outbreaks in a
complex predator-prey system. Through systematic comparison of different
AI configurations (o3-mini and Sonnet-3.5), we demonstrate how our
evolutionary approach can both recover theoretical ecological
relationships and generate practical models for ecosystem management.

# Methods

We developed \"AI for Models of Ecosystems\" (AIME), a framework that
bridges theoretical ecological understanding with practical ecosystem
management by automating the creation and refinement of interpretable
mathematical models. At its core, AIME integrates three key
technologies: Large Language Models (LLMs) for generating and modifying
model structures, Template Model Builder (TMB) for rigorous statistical
parameter estimation, and evolutionary algorithms for systematic model
improvement. The framework requires minimal inputs - only time-series
data and research questions and aims to produce ecologically sound
mathematical models of that explain the time-series data.

To validate AIME's capabilities, we conducted two complementary case
studies. First, we tested the framework's ability to recover fundamental
ecological theory using synthetic data from a
nutrient-phytoplankton-zooplankton (NPZ) model. Second, we evaluated its
practical utility by modeling Crown-of-Thorns starfish populations on
the Great Barrier Reef, comparing our automatically generated models
against expert-developed reference models.

::: landscape
![Conceptual diagram of the automated ecological modeling framework. The
workflow consists of four main components: (1) User Inputs, where
research questions and ecological time-series data are provided; (2)
Parameterisation, utilizing RAG-enhanced literature search to estimate
parameter values; (3) Model Generation/Improvement, where the Coding LLM
creates new individuals with model scripts and parameters; and (4)
Evolution, which evaluates model performance through individual
assessment, error handling, and ranking-based selection. The system
implements comprehensive error checking and parameter validation loops
to ensure model validity before evolutionary
optimization.](conceptual_diagram.png){#fig:conceptual
width="0.85\\linewidth"}
:::

## AIME Framework

### Model Generation and Improvement

AIME uses LLMs to write and modify computer code through a specialized
framework, Aider [@gauthier2024aider]. Each model instance, referred to
as an individual in our evolutionary framework, consists of three
components: (1) a TMB-compatible dynamic model written in C++ that
implements a system of equations, (2) a parameters file containing
initial values and bounds, and (3) a documentation file explaining the
ecological meaning of the model equations (see
Section [7.1](#subsec:initial_model_prompt){reference-type="ref"
reference="subsec:initial_model_prompt"} for the complete prompt).

The LLM generates initial parameter estimates for pre-testing model
structure before optimization begins. For each parameter, it assigns a
priority number that determines optimization order, following
established practices in ecosystem modeling
[@Plaganyi_Punt_Hillary_Morello_Thebaud_Hutton_Pillans_Thorson_Fulton_Smith_et_al_2014].

When improving existing models, if multi-modal (i.e. can receive images
as input) the LLM analyzes performance plots comparing predictions to
historical data, otherwise the LLM receives a structured file showing
the model fit residuals. After interpreting the model fit, AIME makes
targeted, ecologically meaningful changes to model equations,
implementing one modification at a time to maintain transparency and
traceability of successful modeling strategies (see
Section [7.3](#subsec:model_improvement_prompt){reference-type="ref"
reference="subsec:model_improvement_prompt"}).

### Parameterisation

Upon initialization, the LLM estimates parameter values for each
parameter supplied in the model. This initial estimation allows for the
quick subsequent execution of the model, and the rapid discovery of
structural or syntactical errors in the LLM-generated code. If a model
is successful in compiling and running, AIME goes on to find evidence to
support better values for parameters. Building on the success of
LLM-based extraction from ecological literature
[@keck2025extracting; @spillias2024evaluating], the system implements a
Retrieval-Augmented Generation (RAG) architecture to search scientific
literature (see
Section [6](#subsec:rag_architecture){reference-type="ref"
reference="subsec:rag_architecture"} for detailed RAG implementation).
Using this RAG system, an LLM initially creates more detailed semantic
descriptions than those provided by the coding LLM, improving the
likelihood of obtaining relevant search results when searching local
repositories or online databases for parameters. For this simple proof
of concept effort, all parameters, including those with values found
from literature, are treated as estimable parameters in the optimization
process, with literature-derived values, mins and maxes serving as
initial estimates and bounds.

To find appropriate parameter values, the RAG system searches through
scientific literature using three approaches. First, it searches a
database of academic papers called Semantic Scholar, focusing on
highly-cited papers in relevant research fields. Second, it searches
through a carefully selected collection of ecological literature stored
in a local database (see
Section [5](#subsec:curated_literature){reference-type="ref"
reference="subsec:curated_literature"} for the complete collection used
for the CoTS case study). Third, it performs general web searches for
additional information. The system combines results from all three
sources to build a comprehensive understanding of each parameter's
possible values and ecological meaning.

The RAG system uses LLMs to extract numerical values from the search
results, determining not only parameter values but also their valid
ranges (see
Section [7.2](#subsec:parameter_enhancement_prompt){reference-type="ref"
reference="subsec:parameter_enhancement_prompt"}). All parameter
information is stored in a structured JSON database that includes
minimum and maximum bounds, units, and citations to source literature.

### Model Execution and Error Handling

Because LLMs do not consistently return high-quality outputs, we
developed an error handling system to address distinct types of
potential issues. On occasion, the LLM coder will attempt to create a
system of equations with circular logic (which we refer to as 'data
leakage'). Data leakage occurs when the model directly uses observed
values from the current time step to predict those same values, instead
of properly predicting values using only information from previous time
steps. To prevent this, we implement a set of code validation checks to
ensure that the submitted model is properly formatted and free from
logical inconsistencies.

The framework executes models through TMB [@kristensen2014tmb], an
approach which underpins several marine ecosystem modelling frameworks
[@chasco2021differential; @albertsen2015fast; @auger2017spatiotemporal].
TMB provides automatic differentiation techniques for efficient
parameter estimation, with optimization priorities assigned following
established practices
[@Plaganyi_Punt_Hillary_Morello_Thebaud_Hutton_Pillans_Thorson_Fulton_Smith_et_al_2014; @Collie_Botsford_Hastings_Kaplan_Largier_Livingston_Plaganyi_Rose_Wells_Werner_2016; @Rogers_Plaganyi_2022; @Blamey_2022].

For models that pass initial validation, AIME addresses compilation
errors through automated analysis of error messages and implementation
of appropriate fixes. For numerical instabilities, the system employs
progressive simplification of model structure while maintaining
ecological relevance. Each model variant receives up to five iterations
of fixes, with later iterations favoring simpler model structures that
can be iteratively improved. The specific prompts used for error
handling are provided in
Section [7.4](#subsec:error_handling_prompt){reference-type="ref"
reference="subsec:error_handling_prompt"}.

### Model Evaluation

For each response variable $j$, we calculate a normalized mean squared
error:

$$\text{NMSE}_j = \begin{cases}
        \frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_{ij} - \hat{y}_{ij}}{\sigma_j}\right)^2 & \text{if } \sigma_j \neq 0 \\
        \frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2 & \text{if } \sigma_j = 0
    \end{cases}$$

where $y_{ij}$ represents observed values for variable $j$ at time $i$,
$\hat{y}_{ij}$ represents corresponding model predictions, $\sigma_j$ is
the standard deviation of the observed values for variable $j$, and $n$
is the number of observations. The final objective function value is the
mean across all response variables:

$$\text{Objective} = \frac{1}{m} \sum_{j=1}^{m} \text{NMSE}_j$$

where $m$ is the number of response variables. This approach ensures
that each time series contributes equally to the objective function
regardless of its scale or units.

### Evolutionary Algorithm Implementation

The system maintains a population of model instances, which we refer to
as 'individuals', where each individual represents a complete model
implementation including its equations, parameters, and performance
metrics. Within each generation, individuals undergo parameter
optimization using Template Model Builder to find optimal parameter
values for their current model structure.

After parameter optimization, individuals are evaluated based on their
prediction accuracy. Those achieving the lowest prediction errors
(objective values) are selected to become parents for the next
generation, while less well-performing individuals are culled and
non-functioning ones (those that fail to compile or execute) are
discarded.

At the beginning of each new generation, the system creates new
individuals in two ways: by making targeted structural modifications to
the best-performing parent individuals from the previous generation, and
by creating entirely new individuals from scratch when there are not
enough functioning individuals.

## Validation Experiments

We conducted two complementary validation case studies of AIME. The
first validation experiment aimed to see if AIME could recover known
model equations from synthetic time-series data, whilst the second
validation experiment examined real-world applicability through
modelling a set of time-series where the true underlying relationships
were unknown (i.e. were empirically collected data).

### Retrieving Model Equations -- NPZ Case Study

We conducted a controlled experiment using synthetic time-series data
generated by a well-established nutrient-phytoplankton-zooplankton (NPZ)
model from [@edwards1999zooplankton], whose dynamics are well-studied
[@boschetti2008mapping; @boschetti2010detecting]. The complete system of
equations is presented in
Section [7.5](#subsec:npz_evaluation_prompt){reference-type="ref"
reference="subsec:npz_evaluation_prompt"} of the Supplementary
Information. This validation tested our framework's ability to
rediscover established ecological relationships from synthetic data
where the underlying equations of a system are known, providing a
rigorous assessment of the system's equation-learning capabilities.

In addition to monitoring the convergence of AIME towards the provided
time-series data, we evaluated the framework's ability to recover six
key ecological characteristics from the original model, each based on a
discrete term in the system of three equations: nutrient uptake by
phytoplankton with Michaelis-Menten kinetics and self-shading, nutrient
recycling through zooplankton predation and excretion, environmental
mixing of nutrients, phytoplankton growth through nutrient uptake,
phytoplankton losses through mortality and predation, and zooplankton
population dynamics. During evolution, for each 'best performer' in a
generation, we used Claude Sonnet-3.5 to evaluate each model and provide
a score between 0 and 1 for each ecological characteristic based on how
accurately the generated model reproduced the mathematical structure of
the original equations (see
Section [7.5](#subsec:npz_evaluation_prompt){reference-type="ref"
reference="subsec:npz_evaluation_prompt"} for the complete evaluation
prompt). This additional evaluation allowed us to better understand
whether objective value improvements were indeed related to improved
ecological understanding, or whether they were instead related to
spurious mathematical relationships with limited ecological basis. We
ran this evolutionary process using Sonnet-3.5, in four individuals for
60 generations and a convergence threshold of 0.05.

We focused on the framework's ability to recover known ecological
relationships from synthetic data. We analyzed the relationship between
ecological accuracy scores and objective values using two-sided
Pearson's product-moment correlation tests (cor.test in R). For each
ecological characteristic, we calculated the correlation coefficient (r)
and tested the null hypothesis that the true correlation is 0, with the
alternative hypothesis that it is not 0. The resulting p-value indicates
the probability of observing such a correlation by chance if no true
relationship exists, with values below 0.05 considered statistically
significant. This approach allowed us to evaluate whether improvements
in predictive accuracy were achieved through mathematically sound
ecological mechanisms rather than through overfitting.

### Fitting Real World Data -- CoTS Case Study

The Crown-of-Thorns starfish (CoTS) case study examined real-world
applicability through modelling populations of CoTS and their prey,
coral, on the Great Barrier Reef. We tested the leading LLMs (o3-mini
from OpenAI, Claude Sonnet 3.6 and 3.7 from Anthropic, and Gemini-2.0
Flash and Gemini-2.5-pro from Google) within our framework and evaluated
AIME's ability to match the model created by a human expert in the same
context. \*\*\*this needs more elaboration, including citations to
published models, etc.\*\*\*

Due to time and cost constraints we only performed limited tests using
each LLM, where we initialized populations of four individuals for ten
generations each. We decided on these population parameters by balancing
the cost of running each population against the speed of convergence of
a single population that we noted in our initial tests.

We tracked several key performance metrics for each population:

-   Runtime performance: Total runtime and per-generation computation
    time

-   Error resolution: Number of iterations required to achieve
    successful model implementation in each generation

-   Model stability: Proportion of successful, culled (underperforming),
    and numerically unstable models per generation

-   Convergence rate: Improvement in objective value across generations,
    with improvement rate calculated as the change in objective value
    per generation

We also analyzed the evolutionary trajectories of successful models by
tracking their lineage from initial to final states, documenting the
frequency and magnitude of improvements across generations. This
included measuring the number of generations required to reach best
performance and the proportion of attempts that resulted in improved
models.

We also implemented a single evolution where we constructed a temporal
cross-validation approach by partitioning the time series data into
training (pre-2000, approximately 70%) and testing (2000-2005,
approximately 30%) sets. This allowed us to evaluate both in-sample fit
and out-of-sample prediction accuracy. For each ecosystem component
(COTS abundance, fast-growing coral cover, and slow-growing coral
cover), we calculated root mean square error (RMSE), mean absolute error
(MAE), and R² values to quantify prediction accuracy. By comparing these
metrics against those of the human-developed reference model, we could
assess whether our automated approach could match expert-level
performance in a real-world ecological application.

# Results

## Retrieving Model Equations -- NPZ Case Study

Analysis of the NPZ validation study revealed that while AIME did not
perfectly recover the original model equations after 60 generations, it
achieved substantial success in reproducing key ecological mechanisms in
its best-performing models. The top models achieved objective values as
low as 0.0883 while maintaining high ecological accuracy scores (maximum
total score of 7.75 out of 8), demonstrating remarkable accuracy in
reproducing NPZ dynamics. These best performers showed particularly
strong recovery of phytoplankton growth dynamics and zooplankton
equations (scores up to 1.0 for both), demonstrating the framework's
ability to rediscover fundamental ecological relationships. However,
even the best models struggled to recover nutrient mixing terms (maximum
score of 0), suggesting some ecological mechanisms were more challenging
to identify from time-series data alone.

![Evolution and performance of the best NPZ model. (A) Training progress
showing the objective value across generations on a log-scale. (B) Time
series comparison between ground-truth and modelled NPZ dynamics for the
best-performing model (objective value = 0.0883). The plots show the
temporal evolution of nutrient, phytoplankton, and zooplankton
concentrations (g C m^-3^). Blue solid lines represent ground-truth
data, while orange dashed lines show model predictions, demonstrating
the model's ability to capture key ecological patterns and phase
relationships between trophic
levels.](Results/ecology_analysis/NPZ_combined.png){#fig:npz_timeseries
width="\\textwidth"}

Importantly, we found negative correlations between objective values and
ecological accuracy scores, indicating that improvements in model fit
were generally achieved through ecologically sound mechanisms rather
than overfitting. The strongest correlation was observed for
phytoplankton growth equations (r = -0.461, p = 0.002), followed by
nutrient uptake (r = -0.399, p = 0.008). The total ecological score also
showed a significant negative correlation with objective values (r =
-0.380, p = 0.012), suggesting that models achieving better fits tended
to incorporate more correct ecological mechanisms.

The best-performing models achieved objective values as low as 0.112,
demonstrating strong predictive accuracy while maintaining meaningful
ecological structure. A detailed analysis of individual ecological
characteristics (see
Figure [6](#fig:ecological_characteristics){reference-type="ref"
reference="fig:ecological_characteristics"} in Supplementary Materials)
revealed that some mechanisms were more readily recovered than others.
For example, phytoplankton growth dynamics and zooplankton equations
were consistently well-recovered (scores up to 1.0), while nutrient
mixing terms proved more challenging to identify from the time-series
data alone.

![Relationship between total ecological accuracy score and model
performance (objective value). Lower objective values indicate better
model fit, while higher ecological scores indicate closer alignment with
known NPZ model mechanisms. The two-sided Pearson's product-moment
correlation test revealed a significant negative correlation (r =
-0.380, p = 0.012), suggesting that improved model performance was
achieved through recovery of correct ecological relationships rather
than
overfitting.](Results/ecology_analysis/ecological_vs_objective_total.png){#fig:ecological_total
width="80%"}

## Fitting Messy 'Real-World' Data -- CoTS Case Study

For all but one LLM, AIME was able to generate ecosystem models with
prediction accuracy that was comparable to expert-developed models.
After ten generations, analysis across LLM configurations revealed
systematic performance patterns, with o3-mini achieving the best
objective value (0.4213), followed by Claude Sonnet-3.6 (0.4992), Gemini
2.0 Flash (0.5216), and Claude Sonnet-3.7 (0.5808). Surprisingly,
despite high scores on common benchmarks, Google's Gemini-2.5-pro was
not able to produce a single numerically stable model after five
generations, and thus we terminated its process. Component-specific
analysis revealed varying levels of prediction accuracy, with models
showing strongest performance in predicting fast-growing coral cover
(MSE = 0.3015) and slow-growing coral cover (MSE = 0.4486), while
maintaining reasonable accuracy for the more volatile COTS abundance
patterns (MSE = 1.0703).

There were notable differences in the behaviour of convergence for all
of the LLMs we tested. The two Claude LLms were able to generate
well-functioning models in a single generation, but often struggled to
improve upon previous generations, whereas o3-mini (the only 'reasoning'
model we included in this study) was able to consistently improve upon
previous generations. Due to this behaviour, we decided to run o3-mini
for an additional 60 generations (70 generations total), to see if it
would converge at or near the human model. Ultimately, it did reach an
objective value of 0.297, and importantly, it was able to capture the
outbreak dynamics of the CoTS as requested in the research topic Figure
[4](#fig:llm_comparison){reference-type="ref"
reference="fig:llm_comparison"}.

![Comparison of model predictions across ecosystem components. The plots
show observed versus predicted values for COTS abundance, fast-growing
coral cover, and slow-growing coral cover, demonstrating the models'
ability to capture key ecological patterns and relationships. Objective
values (obj) shown in the legend represent the mean normalized mean
squared error across all three variables, where lower values indicate
better model performance. For each variable, the error is normalized by
its observed standard deviation when non-zero, or uses raw squared error
otherwise.](../Figures/llm_predictions_comparison.png){#fig:llm_comparison
width="80%"}

### Time-Series Prediction Performance

Using Claude as the base Large Language Model (LLM), we tested whether
models could be created that are capable of predicting out of sample
datapoints. Quantitative evaluation of out-of-sample predictions
(2000-2005) revealed varying performance across components, with
particularly strong predictive power for slow-growing coral cover (R² =
0.895, RMSE = 2.60, MAE = 1.86).

For fast-growing coral cover, the model achieved moderate predictive
accuracy (RMSE = 2.33, MAE = 1.92, R² = 0.182), effectively capturing
the general declining trend while showing some deviation in precise
values. COTS population predictions demonstrated reasonable accuracy in
absolute terms (RMSE = 0.19, MAE = 0.17) despite a lower R² value
(0.019).

Figure [5](#fig:validation_combined){reference-type="ref"
reference="fig:validation_combined"} illustrates these prediction
capabilities, showing both training period performance (pre-2000) and
out-of-sample predictions (2000-2005). The validation results are
particularly noteworthy given the challenge of simultaneously predicting
multiple interacting ecological components across different temporal
scales. The model's ability to maintain consistent error metrics (RMSE
and MAE) while capturing both rapid population dynamics and slower coral
cover changes suggests it has successfully identified fundamental
ecological relationships governing reef ecosystem dynamics.

![Time-series validation of the best-performing AI model showing
predictions against observed data. The model was trained on pre-2000
data (white background) and validated on unseen 2000-2005 data (pink
shaded region). Top: COTS abundance predictions (RMSE = 0.19, R² =
0.019) showing capture of population variability. Middle: Fast-growing
coral cover predictions (RMSE = 2.33, R² = 0.182) demonstrating tracking
of decline patterns. Bottom: Slow-growing coral cover predictions (RMSE
= 2.60, R² = 0.895) illustrating strong capture of recovery dynamics.
Orange lines represent model predictions, blue dots show observed
data.](../Figures/validation_combined.pdf){#fig:validation_combined
width="60%"}

# Discussion

Our complementary validation studies demonstrate the viability of
AI-driven automation in ecological model development. The NPZ validation
revealed AIME's ability to recover known ecological relationships from
synthetic data, with the best models achieving high ecological accuracy
scores (up to 7.75 out of 8) while maintaining strong predictive
performance (objective values as low as 0.112). While the framework did
not perfectly reconstruct the original equations after 50 generations,
it successfully identified key mechanisms like Michaelis-Menten kinetics
and predator-prey interactions. The negative correlation between
ecological accuracy and objective values suggests that improvements in
model fit were achieved through discovery of correct ecological
relationships rather than overfitting. This ability to balance
mathematical precision with ecological realism represents a significant
advance in automated ecological modeling.

The CoTS case study further demonstrated AIME's practical utility, with
the framework successfully generating models that matched human expert
performance. The achievement of a 43.68% reduction in objective value
from initial conditions, coupled with consistent performance across
multiple populations, suggests that evolutionary algorithms can
effectively navigate the challenging balance between model complexity
and practical utility in real-world applications.

## Contrasting Approaches to AI in Ecological Modeling

Recent advances in AI have demonstrated remarkable capabilities in
ecological time-series prediction. Studies using transformer
architectures and diffusion models, including multimodal approaches like
LITE [@li2024lite], have shown high accuracy in direct forecasting of
environmental variables [@morales2024developing; @gandhi2024generative].
While these methods effectively handle challenges like missing data and
distribution shifts, they treat the system as a black box, learning
patterns directly from time-series data without explicitly modeling
underlying mechanisms. Our NPZ validation study highlights the
limitations of such approaches - while they might achieve strong
predictive performance, they cannot provide insights into fundamental
ecological processes like nutrient cycling or predator-prey dynamics
that our framework explicitly discovers.

Our evolutionary approach fundamentally differs by using AI to generate
actual ecological models rather than make direct predictions. Instead of
training neural networks to forecast future values, AIME evolves
interpretable models with meaningful parameters that capture real
biological and physical processes. This distinction is crucial for
several reasons. First, our generated models provide scientific insight
into system behavior, revealing mechanisms and relationships that direct
prediction approaches cannot. Second, the models maintain biological
plausibility through explicit parameter constraints and mechanistic
formulations, ensuring their utility for management applications. Third,
because they capture fundamental processes rather than just patterns,
these models can potentially be transferred to new scenarios and used to
explore management interventions.

The trade-off between our approach and direct prediction methods is
evident in performance metrics. Recent time-series prediction approaches
using transformer architectures have achieved impressive accuracy, with
mean squared errors as low as 0.001-0.04 for normalized predictions
[@morales2024developing] and root mean squared errors reduced by up to
52% compared to traditional methods [@gandhi2024generative]. While our
evolved models may not always match these pure prediction accuracies,
they offer advantages in interpretability and scientific insight. This
reflects a fundamental tension in ecological modeling between maximizing
predictive performance and maintaining model interpretability. Our
framework demonstrates that it's possible to achieve both reasonable
predictive accuracy and meaningful ecological interpretability, though
pure prediction approaches may achieve marginally better accuracy in
some cases.

This focus on model generation rather than direct prediction aligns with
the needs of ecosystem-based management, where understanding system
dynamics is as important as predictive accuracy. The interpretability of
our evolved models enables managers to assess the credibility of
predictions and understand the mechanisms driving system behavior,
advantages not readily available with black-box prediction approaches.
Recent work examining automated scientific discovery emphasizes the
importance of maintaining human oversight while leveraging AI's
computational capabilities [@kramer2023automated; @Spillias2024]. Our
approach directly addresses this need by producing interpretable models
that facilitate meaningful human oversight while leveraging AI's
capabilities for systematic exploration of model space.

## Implications for Ecosystem-Based Fisheries Management

The successful application of AIME to Crown-of-Thorns starfish
populations on the Great Barrier Reef demonstrates its potential for
informing ecosystem-based management decisions. The framework's capacity
to capture both short-term outbreak dynamics and longer-term ecosystem
changes provides managers with valuable insights for intervention
planning. The comparable performance between AI-generated models and
human expert approaches suggests that automated modeling could
complement traditional methods, potentially accelerating the development
and evaluation of management strategies.

The framework's ability to maintain biological plausibility while
optimizing model performance addresses a critical challenge in ecosystem
modeling. By combining literature-based parameter initialization with
evolutionary refinement, AIME creates models that balance mathematical
precision with ecological realism. This balance is particularly
important for management applications, where model outputs must be both
accurate and interpretable to inform decision-making processes. Our
approach aligns with recent work demonstrating how intermediate
complexity models can effectively capture key ecosystem dynamics while
remaining tractable for management applications [@morello2014model].
Like our study, Morello et al. (2014) found that predator-prey
relationships were critical in regulating population dynamics, with
changes in predator abundance potentially triggering ecosystem regime
shifts. This highlights the importance of considering both direct and
indirect ecological effects when modeling outbreak dynamics.

The success of our modeling framework in capturing complex ecosystem
dynamics while maintaining interpretability suggests it could be
valuable for understanding CoTS outbreaks in other regions. Recent
studies have shown that ecosystem shifts involving CoTS can have
long-lasting effects that are difficult to reverse
[@Pratchett_Caballes_Wilmes_Matthews_Mellin_Sweatman_Nadler_Brodie_Thompson_Hoey_et_al_2017],
making it crucial to detect and respond to outbreaks early. Our model's
ability to integrate multiple data sources and account for both
biological and environmental factors provides a robust foundation for
developing early warning systems and evaluating potential management
interventions.

Despite promising results, several limitations warrant consideration.
The observed variation in convergence rates across populations suggests
that initial conditions significantly influence model evolution
trajectories. While Population 0017 achieved rapid convergence within
five generations, other populations required more than twice as many
generations to approach similar performance levels. This variability
indicates potential opportunities for improving the evolutionary
algorithm's efficiency in exploring parameter space.

An important limitation in our current implementation is the treatment
of all model parameters as estimable quantities in the optimization
process, even when well-established values exist in the literature.
While our RAG system successfully retrieves some literature-based values
and ranges for parameters, these are only used as initial estimates and
bounds rather than as fixed quantities. This approach may lead to
unnecessary parameter estimation and potential deviation from
biologically meaningful values. Future versions of the framework should
distinguish between parameters that truly need estimation and those that
could be fixed based on reliable literature values. This would not only
reduce the parameter space for optimization but also better incorporate
established ecological knowledge into the modeling process.

There are numerous future avenues for validating, improving, and
extending this framework. First, there are several hyper-parameters that
likely control the success and speed of convergence of the framework
(LLM-choice, temperature, number of individuals per generation, prompt
construction, etc.). Systematic testing across these choices may reveal
optimal configurations for convergence. In particular, the comparative
analysis of different AI configurations reveals trade-offs between
computational efficiency and model performance. While the o3-mini
configuration demonstrated faster processing times, the marginally
better performance of the Anthropic Sonnet configuration suggests
potential benefits from more sophisticated language models. Future work
could explore hybrid approaches that leverage the strengths of different
AI configurations at various stages of model development, or that employ
different LLMs consecutively over multiple generations. Further, ongoing
testing of new LLMs as they are released may yield considerable gains in
efficiency and cost-saving. Second, we have tested a relatively simple
ecosystem model with three dependent variable time-series and two
forcing variable time-series. Simple systems like these will be limited
in real-world utility, and therefore testing on more complex systems
with tens or hundreds of time-series will be needed. Whilst the CoTS
case study was aimed at understanding CoTS dynamics, for simplisticity
in this proof-of-concept, we did not weight the time-series in the
objective function, however this might prove useful in future work to
prioritize uncovering key dynamics. Incorporating spatial components may
also be possible and will greatly improve the utility of this framework.
Third, accessing relevant scientific information for the parameter RAG
search is limited by the user's ability to either curate a local
database of relevant materials, or access scientific papers online.
However, due to barriers to accessing scientific papers, this may be
challenging. Fourth, we have demonstrated that it is possible for this
LLM-based system to generate multiple, distinct models for a given
system. Choosing between similarly performing, but ecologically distinct
models may be necessary for experts with ecological knowledge, or
perhaps employing approaches that ensemble multiple plausible models may
allow for the reduction in uncertainty
[@baker2017ensemble; @gaardmark2013biological; @vollert2024unlocking]

# Supplementary Information: An AI-Driven Framework for Automated Generation of Marine Ecosystem Models {#supplementary-information-an-ai-driven-framework-for-automated-generation-of-marine-ecosystem-models .unnumbered}

# Curated Literature Collection {#subsec:curated_literature}

The local document collection used in this case study was carefully
curated to provide comprehensive coverage of marine ecosystem modeling
approaches, with particular focus on COTS-coral dynamics and management
interventions. The collection encompasses several key research areas:

-   Ecosystem Modeling Frameworks: [@Plaganyi_2007] established
    foundational principles for ecosystem approaches to fisheries, while
    [@Plaganyi_Punt_Hillary_Morello_Thebaud_Hutton_Pillans_Thorson_Fulton_Smith_et_al_2014]
    introduced Models of Intermediate Complexity for Ecosystem
    assessments (MICE).
    [@Collie_Botsford_Hastings_Kaplan_Largier_Livingston_Plaganyi_Rose_Wells_Werner_2016]
    explored optimal model complexity levels.

-   COTS Management and Ecology:
    [@Pratchett_Caballes_Wilmes_Matthews_Mellin_Sweatman_Nadler_Brodie_Thompson_Hoey_et_al_2017]
    provided a comprehensive thirty-year review of COTS research.
    [@morello2014model] developed
    models for COTS outbreak management, while [@Rogers_Plaganyi_2022]
    analyzed corallivore culling impacts under bleaching scenarios.

-   Ecological Regime Shifts: [@Blamey_Plaganyi_Branch_2014]
    investigated predator-driven regime shifts in marine ecosystems.
    [@Plaganyi_Ellis_Blamey_Morello_Norman-Lopez_Robinson_Sporcic_Sweatman_2014]
    provided insights into ecological tipping points through ecosystem
    modeling.

-   Management Interventions:
    [@Condie_Anthony_Babcock_Baird_Beeden_Fletcher_Gorton_Harrison_Hobday_Plaganyi_et_al_2021]
    examined large-scale interventions on the Great Barrier Reef.
    [@Punt_MacCall_Essington_Francis_Hurtado-Ferro_Johnson_Kaplan_Koehn_Levin_Sydeman_2016]
    explored harvest control implications using MICE models.

-   Model Application Guidelines: [@Essington_Plaganyi_2014] provided
    critical guidelines for adapting ecosystem models to new
    applications. [@Gamble_Link_2009] demonstrated multispecies
    production model applications for analyzing ecological and fishing
    effects.

-   Integrated Systems: [@Hadley_Wild-Allen_Johnson_Macleod_2015] and
    [@Oca_Cremades_Jimenez_Pintado_Masalo_2019] explored integrated
    multi-trophic aquaculture modeling, providing insights into coupled
    biological systems. [@Spillias_Cottrell_2024] analyzed trade-offs in
    seaweed farming between food production, livelihoods, marine
    biodiversity, and carbon sequestration benefits.

These papers were selected based on their direct relevance to COTS
population dynamics, coral reef ecology, and ecosystem modeling
approaches. The collection provided both specific parameter values and
broader ecological context for model development.

# RAG Architecture Implementation {#subsec:rag_architecture}

The Retrieval-Augmented Generation (RAG) system facilitates parameter
search and extraction from scientific literature. The system employs two
primary search strategies: a local search of user-curated documents and
a comprehensive web search. For local search, the system uses ChromaDB
as a persistent vector store to maintain an indexed collection of
scientific papers and technical documents specifically curated by
research teams for their ecological systems. These documents are
processed into semantic chunks of approximately 512 tokens with small
overlaps to preserve context while enabling precise retrieval of
relevant information.

The parameter search process begins with the generation of enhanced
semantic descriptions for each parameter. These descriptions are crafted
to improve search relevance by capturing the ecological and mathematical
context in which the parameters are used. The system first searches the
user-curated local documents using embeddings generated through Azure
OpenAI's embedding service. When necessary, it extends to web-based
sources through two channels: querying the Semantic Scholar database for
highly-cited papers in biology, mathematics, and environmental science,
and conducting broader literature searches through the Serper API to
capture additional relevant sources.

The search results from both local and web sources are processed through
an LLM to extract numerical values. The system applies consistent
validation across both search pathways, identifying minimum and maximum
bounds, ensuring unit consistency, and validating source reliability.
When direct parameter values are not found in either the local
collection or web sources, the system defaults to the initial estimates
from the coding LLM. All extracted information, including parameter
values, valid ranges, and complete citation details, is stored in a
structured JSON database for reproducibility and future reference.

The RAG system implements automatic retry mechanisms when initial
searches fail to yield usable results. Each retry attempt follows a
structured progression: first accessing the curated local collection
through ChromaDB queries, then expanding to Semantic Scholar for
peer-reviewed literature, and finally utilizing Serper API for broader
scientific content. This progressive broadening of scope, while
maintaining focus on ecologically relevant sources, ensures robust
parameter estimation even in cases where direct measurements are sparse
in the literature.

# AI Prompts Used in Model Development {#sec:ai_prompts}

The development of the model relied on several carefully crafted prompts
to guide the artificial intelligence system. These prompts were designed
to ensure numerical stability, proper likelihood calculation, and clear
model structure. The following sections detail the exact prompts used at
each stage of model development.

## Initial Model Creation {#subsec:initial_model_prompt}

The initial model creation utilized a comprehensive prompt that
emphasized three key aspects of model development. The prompt used for
model initialization was:

    Please create a Template Model Builder model for the following topic:[PROJECT_TOPIC]. Start by writing intention.txt, in which you provide a concise summary of the ecological functioning of the model. In model.cpp, write your TMB model with the following important considerations:

    1. NUMERICAL STABILITY:
    - Always use small constants (e.g., Type(1e-8)) to prevent division by zero
    - Use smooth transitions instead of hard cutoffs in equations
    - Bound parameters within biologically meaningful ranges using smooth penalties rather than hard constraints

    2. LIKELIHOOD CALCULATION:
    - Always include observations in the likelihood calculation, don't skip any based on conditions
    - Use fixed minimum standard deviations to prevent numerical issues when data values are small
    - Consider log-transforming data if it spans multiple orders of magnitude
    - Use appropriate error distributions (e.g., lognormal for strictly positive data)

    3. MODEL STRUCTURE:
    - Include comments after each line explaining the parameters (including their units and how to determine their values)
    - Provide a numbered list of descriptions for the equations
    - Ensure all important variables are included in the reporting section
    - Use `_pred' suffix for model predictions corresponding to `_dat' observations

## Parameter Enhancement {#subsec:parameter_enhancement_prompt}

To enhance parameter descriptions for improved semantic search
capabilities, the following prompt was employed:

    Given a mathematical model about [PROJECT_TOPIC], enhance the semantic descriptions of these parameters to be more detailed and searchable. The model code shows these parameters are used in the following way:

    [MODEL_CONTENT]

    For each parameter below, create an enhanced semantic search, no longer than 10 words, that can be used for RAG search or semantic scholar search.

## Model Improvement {#subsec:model_improvement_prompt}

For iterative model improvements, the system utilized this prompt:

    Improve the fit of the following ecological model by modifying the equations in this TMB script. Only make ONE discrete change most likely to improve the fit. Do not add stochasticity, but you may add other ecological relevant factors that may not be present here already.

    You may add additional parameters if necessary, and if so, add them to parameters.json. Please concisely describe your ecological improvement in intention.txt and then provide the improved model.cpp and parameters.json content.

## Error Handling Prompts {#subsec:error_handling_prompt}

For compilation errors, the system used this prompt:

    model.cpp failed to compile. Here's the error information:

    [ERROR_INFO]

    Do not suggest how to compile the script

For data leakage issues, the system employed this detailed prompt:

    Data leakage detected in model equations. The following response variables cannot be used to predict themselves:

    To fix this:
    1. Response variables ([RESPONSE_VARS]) must be predicted using only:
       - External forcing variables ([FORCING_VARS])
       - Other response variables' predictions (_pred variables)
       - Parameters and constants
    2. Each response variable must have a corresponding prediction equation
    3. Use ecological relationships to determine how variables affect each other

    For example, instead of:
      slow_pred(i) = slow * growth_rate;
    Use:
      slow_pred(i) = slow_pred(i-1) * growth_rate * (1 - impact_rate * cots_pred(i-1));

    Please revise the model equations to avoid using response variables to predict themselves.

For numerical instabilities, the system used an adaptive prompt that
became progressively more focused on simplification after multiple
attempts:

    The model compiled but numerical instabilities occurred. Here's the error information:

    [ERROR_INFO]

    [After 2+ attempts: Consider making a much simpler model that we can iteratively improve later.]
    Do not suggest how to compile the script

## NPZ Case Study - Recovering Equations {#subsec:npz_evaluation_prompt}

The model implementation can be compared to the original NPZ equations
from [@edwards1999zooplankton]:

$$\begin{aligned}
\frac{dN}{dt} &= \underbrace{-\frac{V_m N P}{k_s + N}}_{\text{nutrient uptake}} + \underbrace{\gamma(1-\alpha)\frac{g P^2 Z}{k_g + P^2} + \mu_P P + \mu_Z Z^2}_{\text{recycling}} + \underbrace{S(N_0 - N)}_{\text{mixing}} \\
\frac{dP}{dt} &= \underbrace{\frac{V_m N P}{k_s + N}}_{\text{growth}} - \underbrace{\frac{g P^2 Z}{k_g + P^2} - \mu_P P - S P}_{\text{losses}} \\
\frac{dZ}{dt} &= \underbrace{\alpha\frac{g P^2 Z}{k_g + P^2} - \mu_Z Z^2 - S Z}_{\text{growth and mortality}}
\end{aligned}$$

Our generated model captures several key ecological processes from the
original system:

1.  Nutrient uptake by phytoplankton following Michaelis-Menten kinetics

2.  Quadratic zooplankton mortality

3.  Nutrient recycling through zooplankton excretion

4.  Environmental mixing effects

For evaluating the ecological characteristics of generated models
against the NPZ reference model, the system used this prompt. The prompt
used for all evaluations was:

    Compare this C++ model against the following criteria that should be present in the NPZ model equation by equation.
    The mathematical structure should be identical, even if variable names differ.

    For each equation (dN/dt, dP/dt, dZ/dt), check these components:
    - nutrient_equation_uptake: In dN/dt: Nutrient uptake by phytoplankton with Michaelis-Menten kinetics (N/(e+N)) and self-shading (a/(b+c*P))
    - nutrient_equation_recycling: In dN/dt: Nutrient recycling from zooplankton via predation (beta*lambda*P^2/(mu^2+P^2)*Z) and excretion (gamma*q*Z)
    - nutrient_equation_mixing: In dN/dt: Environmental mixing term (k*(N0-N))
    - phytoplankton_equation_growth: In dP/dt: Phytoplankton growth through nutrient uptake (N/(e+N))*(a/(b+c*P))*P
    - phytoplankton_equation_loss: In dP/dt: Phytoplankton losses through mortality (r*P), predation (lambda*P^2/(mu^2+P^2)*Z), and mixing ((s+k)*P)
    - zooplankton_equation: In dZ/dt: Zooplankton growth through predation (alpha*lambda*P^2/(mu^2+P^2)*Z) and mortality (q*Z)

    For each characteristic:
    1. Score 1.0 if the mathematical structure is equivalent, regardless of:
       - Variable names (e.g., 'N' vs 'nutrients' vs 'N[0]')
       - Programming syntax (e.g., 'pow(x,2)' vs 'x^2')
       - Code organization
    2. Verify the mathematical operations are in the same order
    3. Confirm all coefficients are present (even if named differently)

    Provide your response in JSON format:
    {
        "qualitative_description": "Overall description of how well the model matches the criteria",
        "characteristic_scores": {
            "characteristic_name": {
                "score": 0.0 to 1.0,
                "explanation": "How it is implemented"
            }
        }
    }

To validate the scoring approach, we first tested the evaluation system
on a model containing the exact equations from the original NPZ model.
The system correctly identified and scored all characteristics with
perfect accuracy (score of 1.0 for each component), as shown in this
test evaluation output:

    {
        "qualitative_description": "A nutrient-phytoplankton-zooplankton (NPZ) model with detailed interactions representing nutrient uptake, predation, mortality, and environmental mixing",
        "characteristic_scores": {
            "nutrient_equation_uptake": {
                "score": 1.0,
                "explanation": "Exact Michaelis-Menten kinetics with N/(e+N) and self-shading term a/(b+c*P), implemented identically"
            },
            "nutrient_equation_recycling": {
                "score": 1.0,
                "explanation": "Includes both predation-based recycling (beta*lambda*P^2/(mu^2+P^2)*Z) and zooplankton excretion (gamma*q*Z)"
            },
            "nutrient_equation_mixing": {
                "score": 1.0,
                "explanation": "Environmental mixing term k*(N0-N) present with correct implementation"
            },
            "phytoplankton_equation_growth": {
                "score": 1.0,
                "explanation": "Growth term matches exactly: (N/(e+N))*(a/(b+c*P))*P"
            },
            "phytoplankton_equation_loss": {
                "score": 1.0,
                "explanation": "Includes mortality (r*P), predation (lambda*P^2/(mu^2+P^2)*Z), and mixing ((s+k)*P)"
            },
            "zooplankton_equation": {
                "score": 1.0,
                "explanation": "Zooplankton growth through predation (alpha*lambda*P^2/(mu^2+P^2)*Z) and mortality (q*Z)"
            }
        }
    }

This validation test confirmed that the evaluation system could
correctly identify and score ecological characteristics when present.

# NPZ Validation {#sec:npz_validation}

The NPZ validation study evaluated AIME's ability to recover known
ecological relationships from synthetic data.
Figure [6](#fig:ecological_characteristics){reference-type="ref"
reference="fig:ecological_characteristics"} shows the relationship
between model performance (objective value) and ecological accuracy
scores for each characteristic of the NPZ model. The negative
correlations across multiple characteristics suggest that improvements
in model fit were achieved through discovery of correct ecological
mechanisms rather than overfitting.

![Relationship between ecological accuracy scores and model performance
for each NPZ model characteristic. Each panel shows how well models
recovered a specific ecological mechanism (score from 0-1) versus their
predictive accuracy (objective value). Lower objective values indicate
better model fit. Two-sided Pearson's product-moment correlation
coefficients (r) and their associated p-values are shown for each
characteristic.](Results/ecology_analysis/ecological_characteristics_vs_objective.png){#fig:ecological_characteristics
width="\\textwidth"}

## Best Performing NPZ Model

This model achieved an objective value of 0.0883.

### Model Description

The following model represents our framework's attempt to recover the
NPZ dynamics from [@edwards1999zooplankton]. The model aims to capture
three key components:

-   Nutrient uptake and recycling

-   Phytoplankton growth and mortality

-   Zooplankton predation and dynamics

### Model Intention

    \section{Ecological Intention}

    A key modification was made to incorporate direct nutrient recycling from zooplankton grazing activity. In marine systems, zooplankton feeding is often inefficient, with a significant portion of consumed phytoplankton being released as dissolved nutrients rather than being assimilated into biomass or entering the detritus pool. This "sloppy feeding" process creates an important feedback loop where grazing can stimulate new primary production through rapid nutrient recycling.

    The recycling efficiency is temperature-dependent, reflecting how metabolic rates and feeding mechanics vary with temperature. This creates an adaptive feedback where warmer conditions lead to both increased grazing pressure and faster nutrient recycling, better capturing the coupled nature of predator-prey interactions in planktonic systems.

    The modification introduces a direct pathway from grazing to dissolved nutrients, complementing the slower recycling through the detritus pool. This better represents the multiple timescales of nutrient cycling in marine food webs and helps explain how high productivity can be maintained even under intense grazing pressure.

### Model Implementation

    #include <TMB.hpp>
    template<class Type>
    Type objective_function<Type>::operator() ()
    {
      // Data
      DATA_VECTOR(Time);        // Time points (days)
      DATA_VECTOR(N_dat);       // Nutrient observations (g C m^-3)
      DATA_VECTOR(P_dat);       // Phytoplankton observations (g C m^-3)
      DATA_VECTOR(Z_dat);       // Zooplankton observations (g C m^-3)
      
      // Create default temperature vector if not provided
      vector<Type> Temp(Time.size());
      Temp.fill(Type(20.0));  // Default temperature of 20°C
      
      // Parameters
      PARAMETER(r_max);         // Maximum phytoplankton growth rate (day^-1)
      PARAMETER(K_N);          // Half-saturation constant for nutrient uptake (g C m^-3)
      PARAMETER(g_max);        // Maximum zooplankton grazing rate (day^-1)
      PARAMETER(K_P);          // Half-saturation constant for grazing (g C m^-3)
      PARAMETER(alpha_base);   // Baseline zooplankton assimilation efficiency
      PARAMETER(alpha_max);    // Maximum additional assimilation efficiency
      PARAMETER(K_alpha);      // Half-saturation for nutrient-dependent efficiency
      PARAMETER(m_P);          // Base phytoplankton mortality rate (day^-1)
      PARAMETER(m_P_N);        // Nutrient-dependent phytoplankton mortality (day^-1)
      PARAMETER(s_P);          // Base phytoplankton sinking rate (day^-1)
      PARAMETER(s_P_max);      // Maximum additional nutrient-stress sinking rate (day^-1)
      PARAMETER(m_Z);          // Base zooplankton mortality rate (day^-1)
      PARAMETER(m_Z_N);        // Nutrient-dependent zooplankton mortality (day^-1)
      PARAMETER(r_D);          // Detritus remineralization rate (day^-1)
      PARAMETER(sigma_N);      // SD for nutrient observations
      PARAMETER(sigma_P);      // SD for phytoplankton observations
      PARAMETER(sigma_Z);      // SD for zooplankton observations
      PARAMETER(I_opt);        // Optimal light intensity
      PARAMETER(beta);         // Light attenuation coefficient
      PARAMETER(k_w);         // Light attenuation coefficient due to phytoplankton self-shading
      PARAMETER(E_p);         // Activation energy for photosynthetic efficiency (eV)
      PARAMETER(theta_P);     // Temperature sensitivity of grazing selectivity
      PARAMETER(eta_max);     // Maximum nutrient uptake efficiency multiplier
      PARAMETER(k_eta);       // Steepness of uptake efficiency response
      PARAMETER(N_crit);      // Critical nutrient concentration for efficiency switch
      PARAMETER(eta_base);    // Baseline nutrient uptake efficiency
      

      // Constants for numerical stability
      const Type eps = Type(1e-8);
      const Type min_conc = Type(1e-10);  // Minimum concentration
      const Type max_dt = Type(0.1);      // Maximum time step
      
      // Initialize negative log-likelihood
      Type nll = 0.0;
      
      // Smooth penalties to keep parameters in biological ranges
      nll -= dnorm(log(r_max), Type(0.0), Type(1.0), true);     // Keep r_max positive
      nll -= dnorm(log(K_N), Type(-3.0), Type(1.0), true);      // Keep K_N positive
      nll -= dnorm(log(g_max), Type(-1.0), Type(1.0), true);    // Keep g_max positive
      nll -= dnorm(log(K_P), Type(-3.0), Type(1.0), true);      // Keep K_P positive
      nll -= dnorm(logit(alpha_base), Type(0.0), Type(2.0), true);   // Keep alpha_base between 0 and 1
      nll -= dnorm(logit(alpha_max), Type(0.0), Type(2.0), true);    // Keep alpha_max between 0 and 1
      nll -= dnorm(log(K_alpha), Type(-3.0), Type(1.0), true);       // Keep K_alpha positive
      nll -= dnorm(log(m_P), Type(-3.0), Type(1.0), true);      // Keep m_P positive
      nll -= dnorm(log(m_Z), Type(-3.0), Type(1.0), true);      // Keep m_Z positive
      nll -= dnorm(log(r_D), Type(-3.0), Type(1.0), true);      // Keep r_D positive
      
      // Vectors to store predictions
      vector<Type> N_pred(Time.size());
      vector<Type> P_pred(Time.size());
      vector<Type> Z_pred(Time.size());
      vector<Type> D_pred(Time.size());
      
      // Initial conditions (ensure positive)
      N_pred(0) = exp(log(N_dat(0) + eps));
      D_pred(0) = Type(0.1); // Initial detritus concentration
      P_pred(0) = exp(log(P_dat(0) + eps));
      Z_pred(0) = exp(log(Z_dat(0) + eps));
      
      // Numerical integration using 4th order Runge-Kutta
      for(int t = 1; t < Time.size(); t++) {
        Type dt = Time(t) - Time(t-1);
        
        // Use fixed small time steps for stability
        Type h = Type(0.1); // Fixed step size
        int n_steps = 10;   // Fixed number of steps
        
        Type N = N_pred(t-1);
        Type P = P_pred(t-1);
        Type Z = Z_pred(t-1);
        Type D = D_pred(t-1);
        
        for(int step = 0; step < n_steps; step++) {
          // Temperature scaling (Arrhenius equation)
          Type T_K = Temp(t) + Type(273.15);  // Convert to Kelvin
          Type T_ref = Type(293.15);          // Reference temp (20°C)
          Type E_a = Type(0.63);              // Activation energy (eV)
          Type k_B = Type(8.617e-5);          // Boltzmann constant (eV/K)
          
          // Temperature scaling factor (simplified)
          // General metabolic temperature scaling
          Type temp_scale = exp(E_a * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
          // Photosynthesis-specific temperature scaling
          Type photo_eff = exp(E_p * (Type(1.0)/T_ref - Type(1.0)/T_K) / k_B);
          // Bound scaling factors to prevent numerical issues
          temp_scale = Type(0.5) + Type(0.5) * temp_scale;
          photo_eff = Type(0.5) + Type(0.5) * photo_eff;
          
          // Calculate seasonal light intensity 
          Type season = Type(0.6) * sin(Type(2.0) * M_PI * Time(t) / Type(365.0));
          Type I = I_opt * (Type(1.0) + season);
          
          // Light limitation factor with self-shading
          Type I_effective = I * exp(-k_w * P);  // Reduce light based on phytoplankton density
          Type light_limitation = (I_effective/I_opt) * exp(Type(1.0) - I_effective/I_opt);
          
          // Temperature-dependent grazing selectivity
          Type K_P_T = K_P * (Type(1.0) + theta_P * (temp_scale - Type(1.0)));
          
          // Calculate nutrient-dependent uptake efficiency with baseline
          Type eta_N = eta_base + (eta_max - eta_base) / (Type(1.0) + exp(-k_eta * (N - N_crit)));
          
          // Calculate temperature and light dependent rates
          Type uptake = r_max * temp_scale * photo_eff * light_limitation * eta_N * N * P / (K_N + N + eps);
          
          Type grazing = g_max * temp_scale * P * Z / (K_P_T + P + eps);
          
          // Detritus remineralization (temperature dependent)
          Type remin = r_D * temp_scale * D_pred(t-1);
          
          // System of differential equations
          // Calculate nutrient-dependent assimilation efficiency first
          Type alpha_N = alpha_base + alpha_max * (N / (N + K_alpha + eps));
          
          // Calculate temperature-dependent nutrient recycling efficiency from grazing
          Type recycling_eff = Type(0.3) * temp_scale;  // Base 30% recycling, modified by temperature
          Type grazing_recycle = recycling_eff * (1 - alpha_N) * grazing;
          
          Type dN = -uptake + remin + grazing_recycle;
          
          // Enhanced mortality and sinking under nutrient stress
          Type nutrient_stress = m_P_N * K_N / (N + K_N + eps);
          Type sinking = (s_P + s_P_max * K_N / (N + K_N + eps)) * P;
          Type dP = uptake - grazing - (m_P + nutrient_stress) * P - sinking;
          // Enhanced zooplankton mortality under nutrient stress
          Type Z_nutrient_stress = m_Z_N * K_N / (N + K_N + eps);
          Type dZ = alpha_N * grazing - (m_Z * Z + Z_nutrient_stress) * Z;
          Type dD = m_P * P + m_Z * Z * Z + (1 - alpha_N) * grazing - remin;
          
          // Euler integration step
          N += h * dN;
          P += h * dP;
          Z += h * dZ;
          
          // Ensure concentrations stay positive
          N = exp(log(N + eps));
          P = exp(log(P + eps));
          Z = exp(log(Z + eps));
          D += h * dD;
          D = exp(log(D + eps));
        }
        
        // Store final values
        N_pred(t) = N;
        P_pred(t) = P;
        Z_pred(t) = Z;
        D_pred(t) = D;
      }
      
      // Likelihood calculations using lognormal distribution
      Type min_sigma = Type(0.01);  // Minimum standard deviation
      for(int t = 0; t < Time.size(); t++) {
        nll -= dnorm(log(N_dat(t) + eps), log(N_pred(t) + eps), 
                     exp(log(sigma_N + min_sigma)), true);
        nll -= dnorm(log(P_dat(t) + eps), log(P_pred(t) + eps), 
                     exp(log(sigma_P + min_sigma)), true);
        nll -= dnorm(log(Z_dat(t) + eps), log(Z_pred(t) + eps), 
                     exp(log(sigma_Z + min_sigma)), true);
      }
      
      // Report predictions
      REPORT(N_pred);
      REPORT(P_pred);
      REPORT(Z_pred);
      REPORT(D_pred);

      
      return nll;
    }

### Model Parameters

    {
        "parameters": [
            {
                "parameter": "r_max",
                "value": 1.0,
                "description": "Maximum phytoplankton growth rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Maximum photosynthetic carbon fixation rate in marine ecosystems",
                "citations": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC1913777/",
                    "https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020JG005719",
                    "https://www.sciencedirect.com/science/article/abs/pii/S0967064506001263"
                ],
                "processed": true
            },
            {
                "parameter": "K_N",
                "value": 0.1,
                "description": "Half-saturation constant for nutrient uptake (g C m^-3)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Nutrient limitation threshold for phytoplankton growth dynamics",
                "citations": [
                    "https://www.nature.com/articles/s41467-023-40774-0",
                    "https://www.sciencedirect.com/science/article/pii/S0043135420309428",
                    "https://www.sciencedirect.com/science/article/pii/S1568988325000125"
                ],
                "processed": true
            },
            {
                "parameter": "g_max",
                "value": 0.4,
                "description": "Maximum zooplankton grazing rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Peak predation rate of zooplankton on phytoplankton populations",
                "citations": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3031578/",
                    "https://academic.oup.com/plankt/article/22/6/1085/1587539",
                    "https://www.sciencedirect.com/science/article/abs/pii/S0025556413001466"
                ],
                "processed": true
            },
            {
                "parameter": "K_P",
                "value": 0.2,
                "description": "Half-saturation constant for grazing (g C m^-3)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Prey density threshold controlling zooplankton consumption rates",
                "citations": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9124482/",
                    "https://academic.oup.com/icesjms/article/71/2/254/781831",
                    "https://aslopubs.onlinelibrary.wiley.com/doi/10.1002/lno.10632"
                ],
                "processed": true
            },
            {
                "parameter": "alpha_base",
                "value": 0.2,
                "description": "Baseline zooplankton assimilation efficiency (dimensionless)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Minimum efficiency of energy transfer from prey to zooplankton consumers under nutrient-poor conditions",
                "citations": [
                    "https://link.springer.com/article/10.1007/s10750-017-3298-9",
                    "https://www.sciencedirect.com/science/article/abs/pii/S0022098198000736"
                ],
                "processed": true
            },
            {
                "parameter": "alpha_max",
                "value": 0.3,
                "description": "Maximum additional zooplankton assimilation efficiency (dimensionless)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum additional assimilation efficiency possible under optimal nutrient conditions",
                "citations": [
                    "https://www.sciencedirect.com/science/article/abs/pii/S0022098198000736",
                    "https://www.int-res.com/articles/meps/139/m139p267.pdf"
                ],
                "processed": true
            },
            {
                "parameter": "K_alpha",
                "value": 0.1,
                "description": "Half-saturation constant for nutrient-dependent efficiency (g C m^-3)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Nutrient concentration at which additional assimilation efficiency reaches half maximum",
                "citations": [
                    "https://www.sciencedirect.com/science/article/abs/pii/S0022098198000736"
                ],
                "processed": true
            },
            {
                "parameter": "m_P",
                "value": 0.1,
                "description": "Phytoplankton mortality rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Natural death and senescence rate of phytoplankton communities",
                "citations": [
                    "https://www.sciencedirect.com/science/article/abs/pii/S0079661123001854",
                    "https://www.sciencedirect.com/science/article/abs/pii/S0146638002001067",
                    "https://www.tandfonline.com/doi/full/10.1080/09670262.2018.1563216"
                ],
                "processed": true
            },
            {
                "parameter": "m_Z",
                "value": 0.05,
                "description": "Base zooplankton mortality rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Baseline natural mortality rate of zooplankton populations",
                "citations": [
                    "https://academic.oup.com/icesjms/article/81/6/1164/7697287",
                    "https://www.nature.com/articles/s41558-023-01630-7",
                    "https://www.sciencedirect.com/science/article/pii/S0048969723041281"
                ],
                "processed": true
            },
            {
                "parameter": "m_Z_N",
                "value": 0.1,
                "description": "Nutrient-dependent zooplankton mortality rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Additional zooplankton mortality rate under nutrient-poor conditions",
                "citations": [
                    "https://doi.org/10.4319/lo.2009.54.4.1025",
                    "https://doi.org/10.1016/j.seares.2016.07.002"
                ],
                "processed": true
            },
            {
                "parameter": "r_D",
                "value": 0.1,
                "description": "Detritus remineralization rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Rate at which detrital organic matter is converted back to bioavailable nutrients through bacterial decomposition",
                "citations": [
                    "https://doi.org/10.4319/lo.1992.37.6.1307",
                    "https://doi.org/10.1016/j.marchem.2007.01.006",
                    "https://doi.org/10.1016/j.dsr2.2008.04.029"
                ],
                "processed": true
            },
            {
                "parameter": "sigma_N",
                "value": 0.2,
                "description": "Standard deviation for nutrient observations (log scale)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 4,
                "enhanced_semantic_description": "Measurement uncertainty in marine nutrient concentration observations",
                "processed": true
            },
            {
                "parameter": "sigma_P",
                "value": 0.2,
                "description": "Standard deviation for phytoplankton observations (log scale)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 4,
                "enhanced_semantic_description": "Observational variability in phytoplankton biomass measurements",
                "processed": true
            },
            {
                "parameter": "sigma_Z",
                "value": 0.2,
                "description": "Standard deviation for zooplankton observations (log scale)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 4,
                "enhanced_semantic_description": "Statistical dispersion of zooplankton population density estimates",
                "processed": true
            },
            {
                "parameter": "E_a",
                "value": 0.63,
                "description": "Activation energy for metabolic scaling (eV)",
                "source": "literature",
                "import_type": "CONSTANT",
                "priority": 2,
                "enhanced_semantic_description": "Activation energy controlling temperature dependence of biological rates based on metabolic theory",
                "citations": [
                    "https://doi.org/10.1111/ele.12308",
                    "https://doi.org/10.1126/science.1114383",
                    "https://doi.org/10.1038/nature04095"
                ],
                "processed": true
            },
            {
                "parameter": "I_opt",
                "value": 150.0,
                "description": "Optimal light intensity for photosynthesis (W/m^2)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Optimal irradiance level for phytoplankton photosynthetic efficiency",
                "citations": [
                    "https://doi.org/10.4319/lo.1997.42.7.1552",
                    "https://doi.org/10.1016/j.pocean.2015.04.014"
                ],
                "processed": true
            },
            {
                "parameter": "beta",
                "value": 0.1,
                "description": "Light attenuation coefficient (dimensionless)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Parameter controlling photoinhibition at high light intensities",
                "citations": [
                    "https://doi.org/10.1016/j.ecolmodel.2015.07.014"
                ],
                "processed": true
            },
            {
                "parameter": "m_P_N",
                "value": 0.15,
                "description": "Nutrient-dependent phytoplankton mortality rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Additional phytoplankton mortality rate under nutrient stress conditions",
                "citations": [
                    "https://doi.org/10.1016/j.pocean.2015.04.014",
                    "https://doi.org/10.4319/lo.2009.54.4.1025"
                ],
                "processed": true
            },
            {
                "parameter": "k_w",
                "value": 0.2,
                "description": "Light attenuation coefficient due to phytoplankton self-shading (m^2/g C)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Coefficient describing how phytoplankton biomass reduces light penetration through the water column",
                "citations": [
                    "https://doi.org/10.4319/lo.1990.35.8.1756",
                    "https://doi.org/10.1016/j.ecolmodel.2015.07.014"
                ],
                "processed": true
            },
            {
                "parameter": "E_p",
                "value": 0.45,
                "description": "Activation energy for photosynthetic efficiency (eV)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Activation energy controlling temperature dependence of photosynthetic efficiency, typically lower than general metabolic activation energy due to adaptation of photosynthetic machinery",
                "citations": [
                    "https://doi.org/10.1111/j.1461-0248.2012.01760.x",
                    "https://doi.org/10.1038/nature04095"
                ],
                "processed": true
            },
            {
                "parameter": "s_P",
                "value": 0.15,
                "description": "Base phytoplankton sinking rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Rate at which phytoplankton cells sink out of the surface mixed layer under nutrient-replete conditions",
                "citations": [
                    "https://doi.org/10.4319/lo.1998.43.6.1159",
                    "https://doi.org/10.1016/j.pocean.2015.04.014"
                ],
                "processed": true
            },
            {
                "parameter": "s_P_max",
                "value": 0.3,
                "description": "Maximum additional nutrient-stress sinking rate (day^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum additional sinking rate under severe nutrient limitation as cells become more dense and less buoyant",
                "citations": [
                    "https://doi.org/10.4319/lo.1998.43.6.1159"
                ],
                "processed": true
            },
            {
                "parameter": "eta_max",
                "value": 1.5,
                "description": "Maximum nutrient uptake efficiency multiplier (dimensionless)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum factor by which phytoplankton can enhance their nutrient uptake efficiency under optimal conditions",
                "citations": [
                    "https://doi.org/10.1016/j.pocean.2015.04.014",
                    "https://doi.org/10.4319/lo.2009.54.4.1025"
                ],
                "processed": true
            },
            {
                "parameter": "k_eta",
                "value": 5.0,
                "description": "Steepness of uptake efficiency response (m^3/g C)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Controls how sharply nutrient uptake efficiency changes around the critical nutrient concentration",
                "citations": [
                    "https://doi.org/10.1016/j.pocean.2015.04.014"
                ],
                "processed": true
            },
            {
                "parameter": "N_crit",
                "value": 0.15,
                "description": "Critical nutrient concentration for efficiency switch (g C m^-3)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Nutrient concentration at which uptake efficiency response is centered, representing a physiological threshold",
                "citations": [
                    "https://doi.org/10.4319/lo.2009.54.4.1025"
                ],
                "processed": true
            },
            {
                "parameter": "eta_base",
                "value": 0.5,
                "description": "Baseline phytoplankton nutrient uptake efficiency",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Minimum efficiency of nutrient uptake by phytoplankton under nutrient-limited conditions",
                "citations": [
                    "https://doi.org/10.4319/lo.2009.54.4.1025",
                    "https://doi.org/10.1016/j.pocean.2015.04.014"
                ],
                "processed": true
            },
            {
                "parameter": "theta_P",
                "value": 0.5,
                "description": "Temperature sensitivity of grazing selectivity (dimensionless)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Controls how zooplankton grazing selectivity changes with temperature, reflecting increased selective feeding behavior in warmer conditions",
                "citations": [
                    "https://doi.org/10.1016/j.seares.2018.07.001",
                    "https://doi.org/10.3354/meps09785"
                ],
                "processed": true
            }
        ]
    }

# CoTS Model Convergence

## Model Evolution and Convergence

The evolutionary process demonstrated systematic improvement across
generations, with clear patterns of model refinement and selection. The
mean time to reach best performance was 5.8 generations, with an average
improvement frequency of 41.2% across generations. Figure
[7](#fig:status_distribution){reference-type="ref"
reference="fig:status_distribution"} illustrates the distribution of
successful, culled, and numerically unstable models across generations,
with half of all populations (50%) achieving convergence below the
target threshold.

Generation-by-generation analysis showed varying rates of improvement
across populations. The fastest-converging population reached optimal
performance in just 4 generations, while others required up to 10
generations for refinement. The best-performing population demonstrated
particularly efficient optimization, achieving an objective value of
0.427 within 5 generations and maintaining consistent improvement with a
75% improvement frequency across generations.

![Evolution of model performance during the genetic algorithm
optimization process. Each generation represents an iteration of model
development, where models are evaluated and classified into three
categories: the best performers according to the NMSE objective value
(kept, green), those that are numerically stable, but which are
outcompeted by the best performers (culled, blue), and those whose
scripts threw errors while running, either due to numerical instability,
data leakage, or improper TMB syntax (broken, orange). The vertical axis
shows the count of new models in each category per generation, while
rows represent independent replicates of the optimization process using
different language model configurations (columns). Gemini-2.5-Pro is not
shown here, but was run unsuccessfully for five
generations.](../Figures/success_frequency.png){#fig:status_distribution
width="80%"}

## Numerical Stability and Optimization

The optimization process demonstrated robust numerical stability
characteristics with distinct patterns across LLM configurations. The
o3-mini configuration showed efficient optimization with a mean runtime
of 40.7 minutes and average generation time of 6.0 minutes (SD = 0.86).
In contrast, the anthropic-sonnet configuration required longer
computation times, averaging 99.4 minutes total runtime with 9.9 minutes
per generation (SD = 1.33), though it achieved more consistent final
performance.

AIME employed a phased optimization approach, with iteration counts
varying by population and generation. Analysis of successful model
instances revealed distinct iteration patterns across LLM configurations
(Figure [8](#fig:iterations_by_llm){reference-type="ref"
reference="fig:iterations_by_llm"}). The anthropic-sonnet configuration
demonstrated more consistent iteration requirements, with a median of
2-3 iterations per successful model. In contrast, the o3-mini
configuration showed greater variability but similar efficiency, with
most successful models converging within 2-3 iterations despite some
instances requiring up to 3 iterations. Early generations typically
required more iterations (mean of 6.7 iterations in generation 1), while
later generations showed more efficient convergence (mean of 4.3
iterations in subsequent generations). This pattern was particularly
pronounced in o3-mini populations, which showed rapid convergence to
stable solutions.

![Distribution of iteration counts for successful model instances by LLM
configuration. The boxplot shows the number of iterations required for
convergence, excluding cases that reached maximum iterations or remained
numerically
unstable.](../Figures/iterations_by_llm.png){#fig:iterations_by_llm
width="80%"}

Model evolution trajectories exhibited systematic improvement in both
stability and accuracy, as illustrated in Figure
[\[fig:evolution\]](#fig:evolution){reference-type="ref"
reference="fig:evolution"}. The mean improvement rate across populations
was -0.037 objective value units per generation, with the
best-performing population achieving a notably rapid improvement rate of
-0.0575. AIME maintained efficient exploration of the parameter space,
with mean generation times of 7.3 minutes (SD = 2.2) across all
populations.

# Best Performing Models for CoTS Case Study {#sec:best_models}

This section presents the 4 best performing models from different
configurations for the Crown of Thorns Starfish case study.

## o3 mini Model

This model achieved an objective value of 0.2971.

### Model Implementation

    /*
    Equations description:
    1. COTS dynamics:
       cots_pred(t) = cots_pred(t-1) + [r_COTS * cots_pred(t-1) * ( (slow_pred(t-1)+fast_pred(t-1)) / (half_sat + slow_pred(t-1)+fast_pred(t-1) + 1e-8) ) * env - m_COTS * cots_pred(t-1)]
       - r_COTS: reproduction rate (year^-1)
       - m_COTS: mortality rate (year^-1)
       - env: environmental modifier (unitless)
    2. Slow coral dynamics:
       slow_pred(t) = slow_pred(t-1) + growth_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow)
                       - (cots_pred(t-1)*slow_pred(t-1))/(half_sat + slow_pred(t-1) + 1e-8)
       - growth_slow: intrinsic growth rate (year^-1)
       - K_slow: carrying capacity (units corresponding to coral cover)
    3. Fast coral dynamics:
       fast_pred(t) = fast_pred(t-1) + growth_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast)
                       - (cots_pred(t-1)*fast_pred(t-1))/(half_sat + fast_pred(t-1) + 1e-8)
       - growth_fast: intrinsic growth rate (year^-1)
       - K_fast: carrying capacity
    Numerical constants (1e-8) are added to avoid division by zero.
    Only past time-step values are used in predictions to prevent data leakage.
    */

    #include <TMB.hpp>
    #include <algorithm>

    template<class Type>
    Type objective_function<Type>::operator() ()
    {
      // Data inputs: each DATA_VECTOR should match the column names from the observations data file.
      DATA_VECTOR(Year);       // Year [integer]
      DATA_VECTOR(cots_dat);     // Observed COTS density (indiv/m^2)
      DATA_VECTOR(slow_dat);     // Observed slow coral cover (Faviidae/Porites, in %)
      DATA_VECTOR(fast_dat);     // Observed fast coral cover (Acropora spp., in %)
      DATA_VECTOR(sst_dat);      // Sea-surface temperature in Celsius
      DATA_VECTOR(cotsimm_dat);  // COTS larval immigration rate (indiv/m^2/year)

      int n = Year.size();
      
      // Model parameters (all using log-transformed values for stability)
      PARAMETER(log_r_COTS);      // Log reproduction rate for COTS (year^-1), from literature or estimation
      PARAMETER(log_m_COTS);      // Log mortality rate for COTS (year^-1)
      PARAMETER(log_growth_slow); // Log intrinsic growth rate for slow coral (year^-1)
      PARAMETER(log_growth_fast); // Log intrinsic growth rate for fast coral (year^-1)
      PARAMETER(log_K_slow);      // Log carrying capacity for slow coral (coral cover units)
      PARAMETER(log_K_fast);      // Log carrying capacity for fast coral (coral cover units)
      PARAMETER(log_half_sat);    // Log half-saturation constant for coral predation effect (matching coral cover units)
      PARAMETER(log_env);         // Log environmental modifier for COTS reproduction (unitless)
      PARAMETER(log_sst_sensitivity); // Log sensitivity of COTS reproduction to previous sea-surface temperature anomaly.
      PARAMETER(log_coral_temp_sensitivity); // Log sensitivity of coral growth rate to temperature deviations (optimal growth at opt_temp_coral)
      PARAMETER(opt_temp_coral); // Optimal sea-surface temperature for coral growth.
      PARAMETER(log_cots_temp_sensitivity); // Log sensitivity of COTS reproduction to temperature deviations
      PARAMETER(opt_temp_COTS); // Optimal sea-surface temperature for triggering COTS reproductive outbreak.
      PARAMETER(log_temp_skew); // Log skew parameter for COTS temperature sensitivity asymmetry.
      PARAMETER(log_temp_poly);
      PARAMETER(log_beta); // Log cross-species competition coefficient; exp(log_beta) represents interspecific competition between slow and fast corals.
      PARAMETER(log_damage_slow); // Log damage scaling parameter for slow coral due to cumulative predation effects.
      PARAMETER(log_allee_threshold); // Log Allee threshold for COTS reproduction (ecological mate-finding limitation)
      PARAMETER(log_self_limiting_COTS); // Log self-limiting term for COTS density dependence.
      PARAMETER(log_pred_exponent); // Log exponent for flexible predation response on coral
      PARAMETER(log_half_sat_pred); // Log half-saturation constant for predation on coral; independent parameter
      
      // Observation error parameters (log-transformed to ensure positivity)
      PARAMETER(log_sd_COTS);     // Log standard deviation for COTS observations
      PARAMETER(log_sd_slow);     // Log standard deviation for slow coral observations
      PARAMETER(log_sd_fast);     // Log standard deviation for fast coral observations

      // Transform parameters from log scale
      Type r_COTS    = exp(log_r_COTS);
      Type m_COTS    = exp(log_m_COTS);
      Type growth_slow = exp(log_growth_slow);
      Type growth_fast = exp(log_growth_fast);
      Type K_slow    = exp(log_K_slow);
      Type K_fast    = exp(log_K_fast);
      Type half_sat  = exp(log_half_sat);
      Type half_sat_pred = exp(log_half_sat_pred);
      Type env       = exp(log_env);
      Type sst_sensitivity = exp(log_sst_sensitivity);
      Type coral_temp_sensitivity = exp(log_coral_temp_sensitivity);
      Type sd_COTS   = exp(log_sd_COTS) + Type(1e-8);
      Type sd_slow   = exp(log_sd_slow) + Type(1e-8);
      Type sd_fast   = exp(log_sd_fast) + Type(1e-8);
      Type cots_temp_sensitivity = exp(log_cots_temp_sensitivity);
      Type temp_skew = exp(log_temp_skew);
      Type temp_poly = exp(log_temp_poly);
      Type beta = exp(log_beta);
      Type damage_slow = exp(log_damage_slow);
      Type self_limiting_COTS = exp(log_self_limiting_COTS);
      Type pred_exponent = std::min(exp(log_pred_exponent), Type(10.0));
      Type allee_threshold = exp(log_allee_threshold);

      // Vectors to store model predictions
      vector<Type> cots_pred(n);
      vector<Type> slow_pred(n);
      vector<Type> fast_pred(n);

      // Initialize predictions with the first observation (acting as the initial condition)
      cots_pred(0) = cots_dat(0);
      slow_pred(0) = slow_dat(0);
      fast_pred(0) = fast_dat(0);

      Type nll = 0.0;
      // Loop through time steps (starting from t=1; we only use previous time-step values)
      for(int t = 1; t < n; t++){
        // Equation 1: COTS dynamics
        Type coral_total = slow_pred(t-1) + fast_pred(t-1);
        // Coral modifier: saturating effect of available coral cover on reproduction
        Type coral_modifier = (pow(coral_total, 2)) / (pow(half_sat, 2) + pow(coral_total, 2) + Type(1e-8));
        // Reproduction term with Gaussian temperature effect for COTS reproduction dynamics
        Type deviation = sst_dat(t-1) - opt_temp_COTS;
        Type temp_effect_COTS = (sst_dat(t-1) > opt_temp_COTS) ? exp(-cots_temp_sensitivity * temp_skew * deviation * deviation)
                                                              : exp(-cots_temp_sensitivity * deviation * deviation);
        Type allee_effect = 1.0 / (1.0 + exp(-10 * (cots_pred(t-1) - allee_threshold)));
        Type reproduction = r_COTS * cots_pred(t-1) * coral_modifier * env * temp_effect_COTS * allee_effect;
        cots_pred(t) = cots_pred(t-1) + ( reproduction - m_COTS * cots_pred(t-1) - self_limiting_COTS * cots_pred(t-1) * cots_pred(t-1) );
        if(cots_pred(t) < Type(1e-8)) { cots_pred(t) = Type(1e-8); }

        // Equation 2: Slow coral dynamics with logistic growth modulated by temperature and COTS predation (Type III response)
        Type temp_multiplier = exp(-coral_temp_sensitivity * (sst_dat(t-1) - opt_temp_coral) * (sst_dat(t-1) - opt_temp_coral) * (1 + temp_poly * fabs(sst_dat(t-1) - opt_temp_coral)));
        Type predation_slow = (cots_pred(t-1) * pow(slow_pred(t-1), pred_exponent)) / (pow(half_sat_pred, pred_exponent) + pow(slow_pred(t-1), pred_exponent) + Type(1e-8));
        slow_pred(t) = slow_pred(t-1) + growth_slow * slow_pred(t-1) * temp_multiplier * (1 - slow_pred(t-1) / K_slow) - predation_slow - beta * slow_pred(t-1) * fast_pred(t-1) / K_slow - damage_slow * slow_pred(t-1) * pow(cots_pred(t-1) / (K_slow + Type(1e-8)), 2);
        if(slow_pred(t) < Type(1e-8)) { slow_pred(t) = Type(1e-8); }

        // Equation 3: Fast coral dynamics with logistic growth modulated by temperature and COTS predation (Type III response)
        Type temp_multiplier_fast = exp(-coral_temp_sensitivity * (sst_dat(t-1) - opt_temp_coral) * (sst_dat(t-1) - opt_temp_coral) * (1 + temp_poly * fabs(sst_dat(t-1) - opt_temp_coral)));
        Type predation_fast = (cots_pred(t-1) * pow(fast_pred(t-1), pred_exponent)) / (pow(half_sat_pred, pred_exponent) + pow(fast_pred(t-1), pred_exponent) + Type(1e-8));
        fast_pred(t) = fast_pred(t-1) + growth_fast * fast_pred(t-1) * temp_multiplier_fast * (1 - fast_pred(t-1) / K_fast) - predation_fast - beta * slow_pred(t-1) * fast_pred(t-1) / K_fast;
        if(fast_pred(t) < Type(1e-8)) { fast_pred(t) = Type(1e-8); }

        // Likelihood: assuming observations come from a normal distribution around model predictions
        nll -= dnorm(cots_dat(t), cots_pred(t), sd_COTS, true);
        nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
        nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
      }

      // REPORT predictions so that they can be output and inspected
      REPORT(cots_pred);
      REPORT(slow_pred);
      REPORT(fast_pred);

      return nll;
    }

### Model Parameters

    {
        "parameters": [
            {
                "parameter": "log_r_COTS",
                "value": -0.6931,
                "description": "Log reproduction rate for COTS (year^-1); exp(log_r_COTS) gives reproduction rate.",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown of Thorns starfish population reproduction rate dynamics",
                "citations": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9023020/",
                    "https://www.cell.com/current-biology/pdf/S0960-9822(13)00969-X.pdf",
                    "https://newheavenreefconservation.org/projects/crown-of-thorns"
                ],
                "processed": true
            },
            {
                "parameter": "log_m_COTS",
                "value": -2.3026,
                "description": "Log mortality rate for COTS (year^-1); exp(log_m_COTS) gives mortality rate.",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown of Thorns starfish mortality and population decline mechanisms",
                "processed": true
            },
            {
                "parameter": "log_growth_slow",
                "value": -1.6094,
                "description": "Log intrinsic growth rate for slow coral (year^-1); exp(log_growth_slow) gives growth rate.",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Slow-growing coral species intrinsic growth and recovery potential",
                "processed": true
            },
            {
                "parameter": "log_growth_fast",
                "value": -1.2039,
                "description": "Log intrinsic growth rate for fast coral (year^-1); exp(log_growth_fast) gives growth rate.",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Fast-growing coral species resilience and regeneration capacity",
                "processed": true
            },
            {
                "parameter": "log_K_slow",
                "value": 3.912,
                "description": "Log carrying capacity for slow coral; exp(log_K_slow) gives the carrying capacity (coral cover units).",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Slow coral ecosystem maximum sustainable population and habitat capacity",
                "processed": true
            },
            {
                "parameter": "log_K_fast",
                "value": 4.2485,
                "description": "Log carrying capacity for fast coral; exp(log_K_fast) gives the carrying capacity (coral cover units).",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Fast coral ecosystem maximum sustainable population and habitat capacity",
                "processed": true
            },
            {
                "parameter": "log_half_sat",
                "value": 2.3026,
                "description": "Log half-saturation constant for coral predation effects; exp(log_half_sat) gives the threshold used in the sigmoidal (Type III) response.",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Sigmoidal functional response threshold for coral predation, capturing low predation efficiency at low coral densities and heightened predation as coral cover increases",
                "citations": [
                    "https://www.pnas.org/doi/10.1073/pnas.1106861108",
                    "https://www.sciencedirect.com/science/article/abs/pii/S0022519324003163",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8488628/"
                ],
                "processed": true
            },
            {
                "parameter": "log_env",
                "value": 0,
                "description": "Log environmental modifier (unitless) affecting COTS reproduction efficiency; exp(log_env) gives multiplier.",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Environmental factors influencing Crown of Thorns starfish reproduction",
                "processed": true
            },
            {
                "parameter": "log_sd_COTS",
                "value": -2.3026,
                "description": "Log standard deviation for COTS observations; exp(log_sd_COTS) gives standard deviation.",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Measurement uncertainty and variability in Crown of Thorns observations",
                "processed": true
            },
            {
                "parameter": "log_sd_slow",
                "value": -2.3026,
                "description": "Log standard deviation for slow coral observations; exp(log_sd_slow) gives standard deviation.",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Measurement uncertainty in slow-growing coral population assessments",
                "processed": true
            },
            {
                "parameter": "log_sd_fast",
                "value": -2.3026,
                "description": "Log standard deviation for fast coral observations; exp(log_sd_fast) gives standard deviation.",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Measurement uncertainty in fast-growing coral population assessments",
                "processed": true
            },
            {
                "parameter": "log_sst_sensitivity",
                "value": 0,
                "description": "Log sensitivity of COTS reproduction to previous sea-surface temperature anomaly; exp(log_sst_sensitivity) multiplies the deviation from a 26\u00c2\u00b0C baseline.",
                "source": "hypothesis/model improvement",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Modulates the effect of sea-surface temperature anomalies on COTS reproduction rates, capturing episodic outbreak dynamics",
                "processed": true
            },
            {
                "parameter": "log_coral_temp_sensitivity",
                "value": -1.0,
                "description": "Log sensitivity of coral growth rate to deviations in sea-surface temperature from the optimum (26\u00c2\u00b0C).",
                "source": "hypothesis/model improvement",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Captures reduced coral growth when temperatures deviate from the optimum (26\u00c2\u00b0C).",
                "processed": true
            },
            {
                "parameter": "opt_temp_coral",
                "value": 26.0,
                "description": "Optimal sea-surface temperature for coral growth.",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Represents the temperature at which coral growth is maximized.",
                "processed": true
            },
            {
                "parameter": "log_cots_temp_sensitivity",
                "value": -1.0,
                "description": "Log sensitivity of COTS reproduction to deviations in sea-surface temperature from the optimum (opt_temp_COTS).",
                "source": "hypothesis/model improvement",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Captures the non-linear effect of temperature on COTS reproduction, triggering outbreak events when optimal conditions are met.",
                "processed": true
            },
            {
                "parameter": "opt_temp_COTS",
                "value": 28.0,
                "description": "Optimal sea-surface temperature for triggering COTS reproductive outbreak.",
                "source": "hypothesis/model improvement",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Represents the temperature at which COTS reproduction is maximized, contributing to outbreak events.",
                "processed": true
            },
            {
                "parameter": "log_self_limiting_COTS",
                "value": -3.0,
                "description": "Log self-limiting term for COTS, representing density-dependent intraspecific competition that dampens outbreak magnitude at high population levels.",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Models increased competition and resource limitation in high-density COTS populations, reducing explosive growth.",
                "processed": true
            },
            {
                "parameter": "log_allee_threshold",
                "value": -3.0,
                "description": "Log threshold for Allee effect in COTS reproduction; exp(log_allee_threshold) gives the minimum COTS density required for effective mate-finding and reproduction.",
                "source": "ecological theory",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Represents a critical density threshold below which COTS reproduction is limited, capturing mate-finding Allee effects that trigger outbreak events above threshold.",
                "processed": true
            },
            {
                "parameter": "log_pred_exponent",
                "value": 0.6931,
                "description": "Log flexible exponent for predation response on coral; exp(log_pred_exponent) adjusts sensitivity of predation to coral cover, capturing threshold effects.",
                "source": "model refinement",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Allows the predation rate on coral to vary non-linearly with coral cover, enabling better capture of threshold dynamics in COTS outbreaks.",
                "processed": true
            },
            {
                "parameter": "log_half_sat_pred",
                "value": 2.3026,
                "description": "Log half-saturation constant for predation on coral; exp(log_half_sat_pred) defines the coral density at which COTS predation efficiency saturates, independent from reproduction dynamics.",
                "source": "model improvement",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Introduces a separate saturating effect for the COTS predation functional response, allowing independent tuning of coral predation dynamics during outbreaks.",
                "processed": true
            },
            {
                "parameter": "log_temp_skew",
                "value": 0,
                "description": "Log skew parameter for COTS temperature sensitivity asymmetry. Values >0 indicate a sharper decline in reproduction with temperatures above the optimum relative to below.",
                "source": "model improvement",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Introduces asymmetry in the temperature-dependent reproduction of COTS, capturing differential effects of temperature deviations above vs below the optimum",
                "processed": true
            },
            {
                "parameter": "log_temp_poly",
                "value": -1.2039,
                "description": "Log parameter for additional non-linear sensitivity of coral growth to temperature deviations. exp(log_temp_poly) scales the effect of temperature deviation on reducing coral growth, capturing sharper declines during extreme thermal events.",
                "source": "ecological studies",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Introduces a polynomial modifier to the coral temperature sensitivity, enhancing the decline in growth as deviation from optimal increases",
                "processed": true
            },
            {
                "parameter": "log_beta",
                "value": -2.3026,
                "description": "Log cross-species competition coefficient; exp(log_beta) represents the strength of interspecific competition (resource limitation) between slow and fast corals.",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Represents the competition for space and nutrients between slow and fast coral, influencing system dynamics by indirectly modulating predation on corals during COTS outbreaks.",
                "processed": true
            },
            {
                "parameter": "log_damage_slow",
                "value": -2.0,
                "description": "Log damage scaling parameter for slow coral due to cumulative predation effects; lower values imply higher damage effects. Now applied non-linearly as the squared ratio of COTS density to coral carrying capacity, capturing catastrophic coral loss at high COTS levels.",
                "source": "model improvement",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Introduces a non-linear feedback mechanism for cumulative tissue damage in slow coral by squaring the relative COTS density, better capturing the threshold behavior and catastrophic coral loss during outbreaks.",
                "processed": true
            }
        ]
    }

## anthropic sonnet Model

This model achieved an objective value of 0.4992.

### Model Intention

    The model has been enhanced to include a more realistic representation of Crown-of-Thorns Starfish (COTS) population dynamics by incorporating resource-dependent reproduction. In the original model, COTS population growth was linked to predation but didn't explicitly account for how coral abundance affects COTS reproduction.

    The new formulation includes a resource-dependent reproduction term where COTS population growth is directly tied to coral consumption through the parameter \beta (beta). This better represents the biological reality that COTS require adequate coral tissue consumption to fuel reproduction. The reproduction term is calculated as:

    reproduction = \beta * (f_slow + f_fast) * total_cover

    where:
    - \beta represents the efficiency of converting consumed coral tissue into new COTS individuals
    - (f_slow + f_fast) captures the total predation rate
    - total_cover scales the reproduction based on available coral resources

    This modification creates a more realistic feedback loop between coral abundance and COTS population dynamics, which is crucial for capturing outbreak cycles. When coral cover is high, COTS reproduction increases, potentially leading to outbreaks. As corals are depleted, COTS reproduction decreases, allowing coral recovery - a pattern observed in real reef systems.

### Model Implementation

    #include <TMB.hpp>
    template<class Type>
    Type objective_function<Type>::operator() ()
    {
      // Data inputs
      DATA_VECTOR(Year);           // Time vector (years)
      DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m$^2$)
      DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
      DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
      DATA_VECTOR(sst_dat);        // Sea surface temperature (°C)
      DATA_VECTOR(cotsimm_dat);    // COTS immigration rate (individuals/m$^2$/year)
      
      // Parameters
      PARAMETER(log_r_slow);       // Log of slow coral intrinsic growth rate
      PARAMETER(log_r_fast);       // Log of fast coral intrinsic growth rate
      PARAMETER(log_K_slow);       // Log of slow coral carrying capacity
      PARAMETER(log_K_fast);       // Log of fast coral carrying capacity
      PARAMETER(log_alpha_slow);   // Log of COTS attack rate on slow coral
      PARAMETER(log_alpha_fast);   // Log of COTS attack rate on fast coral
      PARAMETER(log_h_slow);       // Log of handling time for slow coral
      PARAMETER(log_h_fast);       // Log of handling time for fast coral
      PARAMETER(log_m);            // Log of COTS density-dependent mortality
      PARAMETER(log_T_opt);        // Log of optimal temperature for COTS
      PARAMETER(log_sigma_T);      // Log of temperature tolerance width
      PARAMETER(log_obs_sd);       // Log of observation error SD
      PARAMETER(log_beta);         // Log of COTS reproduction efficiency
      
      // Transform parameters
      Type r_slow = exp(log_r_slow);
      Type r_fast = exp(log_r_fast);
      Type K_slow = exp(log_K_slow);
      Type K_fast = exp(log_K_fast);
      Type alpha_slow = exp(log_alpha_slow);
      Type alpha_fast = exp(log_alpha_fast);
      Type h_slow = exp(log_h_slow);
      Type h_fast = exp(log_h_fast);
      Type m = exp(log_m);
      Type T_opt = exp(log_T_opt);
      Type sigma_T = exp(log_sigma_T);
      Type obs_sd = exp(log_obs_sd);
      Type beta = exp(log_beta);
      
      // Initialize negative log-likelihood
      Type nll = 0.0;
      
      // Initialize vectors for predictions
      vector<Type> cots_pred(Year.size());
      vector<Type> slow_pred(Year.size());
      vector<Type> fast_pred(Year.size());
      vector<Type> cotsimm_pred(Year.size());
      
      // Small constant to prevent division by zero
      Type eps = Type(1e-8);
      
      // Initial conditions
      cots_pred(0) = cots_dat(0);
      slow_pred(0) = slow_dat(0);
      fast_pred(0) = fast_dat(0);
      cotsimm_pred(0) = cotsimm_dat(0);
      
      // Process model
      for(int t = 1; t < Year.size(); t++) {
        // Temperature stress effects - different for each coral type
        Type temp_stress_fast = exp(-2.0 * pow((sst_dat(t) - T_opt) / sigma_T, 2));
        Type temp_stress_slow = exp(-0.5 * pow((sst_dat(t) - T_opt) / sigma_T, 2));
        
        // Total coral cover with temperature-modified competition
        Type total_cover = slow_pred(t-1) + fast_pred(t-1);
        
        // Temperature-dependent predation efficiency
        Type pred_efficiency = exp(-0.5 * pow((sst_dat(t) - T_opt) / (sigma_T * 1.5), 2));
        
        // Holling Type II functional responses with temperature-dependent attack rates
        Type f_slow = (alpha_slow * pred_efficiency * slow_pred(t-1)) / 
                     (1 + alpha_slow * pred_efficiency * h_slow * slow_pred(t-1) + 
                      alpha_fast * pred_efficiency * h_fast * fast_pred(t-1));
        Type f_fast = (alpha_fast * pred_efficiency * fast_pred(t-1)) / 
                     (1 + alpha_slow * pred_efficiency * h_slow * slow_pred(t-1) + 
                      alpha_fast * pred_efficiency * h_fast * fast_pred(t-1));
        
        // COTS dynamics with coral-dependent reproduction
        Type temp_effect_cots = exp(-0.5 * pow((sst_dat(t) - T_opt) / sigma_T, 2));
        cotsimm_pred(t) = temp_effect_cots * std::max(Type(0), cotsimm_pred(t-1));
        
        // Resource-dependent reproduction based on coral consumption
        // Scale reproduction by relative coral abundance to prevent unrealistic growth
        Type reproduction = beta * (f_slow + f_fast) * total_cover / (total_cover + K_slow + K_fast);
        
        cots_pred(t) = cots_pred(t-1) + 
                       temp_effect_cots * reproduction * cots_pred(t-1) -
                       m * pow(cots_pred(t-1), 2) +
                       cotsimm_pred(t);
        cots_pred(t) = std::max(cots_pred(t), eps);
        
        // Coral dynamics with space limitation
        slow_pred(t) = slow_pred(t-1) + 
                       r_slow * slow_pred(t-1) * (1 - total_cover/K_slow) * temp_stress_slow -
                       f_slow * cots_pred(t-1);
        slow_pred(t) = std::max(slow_pred(t), eps);
        
        fast_pred(t) = fast_pred(t-1) + 
                       r_fast * fast_pred(t-1) * (1 - total_cover/K_fast) * temp_stress_fast -
                       f_fast * cots_pred(t-1);
        fast_pred(t) = std::max(fast_pred(t), eps);
      }
      
      // Observation model (lognormal)
      for(int t = 0; t < Year.size(); t++) {
        nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), obs_sd, true);
        nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), obs_sd, true);
        nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), obs_sd, true);
      }
      
      // Report predictions
      REPORT(cots_pred);
      REPORT(slow_pred);
      REPORT(fast_pred);
      REPORT(cotsimm_pred);
      
      // Report objective function value
      ADREPORT(nll);
      
      return nll;
    }

### Model Parameters

    {
        "parameters": [
            {
                "parameter": "log_r_slow",
                "value": -1.6094,
                "description": "Log of slow-growing coral intrinsic growth rate (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Slow-growing coral population regeneration and recovery potential",
                "citations": [
                    "https://pubmed.ncbi.nlm.nih.gov/39106878/",
                    "https://oceanservice.noaa.gov/ocean/corals/hope-for-corals.html",
                    "https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.4915"
                ],
                "processed": true
            },
            {
                "parameter": "log_r_fast",
                "value": -0.9163,
                "description": "Log of fast-growing coral intrinsic growth rate (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Fast-growing coral population resilience and expansion rate",
                "citations": [
                    "https://oceanservice.noaa.gov/ocean/corals/hope-for-corals.html",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9545251/",
                    "https://www.nature.com/articles/s41467-022-30234-6"
                ],
                "processed": true
            },
            {
                "parameter": "log_K_slow",
                "value": 3.912,
                "description": "Log of slow-growing coral carrying capacity (%)",
                "source": "data maximum",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum sustainable population size for slow-growing coral species",
                "processed": true
            },
            {
                "parameter": "log_K_fast",
                "value": 3.4012,
                "description": "Log of fast-growing coral carrying capacity (%)",
                "source": "data maximum",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum sustainable population size for fast-growing coral species",
                "processed": true
            },
            {
                "parameter": "log_alpha_slow",
                "value": -1.6094,
                "description": "Log of COTS attack rate on slow coral (m\u00c2\u00b2/ind/year)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown-of-thorns starfish predation intensity on slow-growing corals",
                "processed": true
            },
            {
                "parameter": "log_alpha_fast",
                "value": -0.6931,
                "description": "Log of COTS attack rate on fast coral (m\u00c2\u00b2/ind/year)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown-of-thorns starfish predation intensity on fast-growing corals",
                "processed": true
            },
            {
                "parameter": "log_h_slow",
                "value": -2.3026,
                "description": "Log of handling time for slow coral (year)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Feeding consumption time for predation on slow-growing coral species",
                "processed": true
            },
            {
                "parameter": "log_h_fast",
                "value": -2.9957,
                "description": "Log of handling time for fast coral (year)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Feeding consumption time for predation on fast-growing coral species",
                "processed": true
            },
            {
                "parameter": "log_m",
                "value": -0.6931,
                "description": "Log of COTS density-dependent mortality (m\u00c2\u00b2/ind/year)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Crown-of-thorns starfish population self-regulation and mortality rate",
                "processed": true
            },
            {
                "parameter": "log_T_opt",
                "value": 3.2581,
                "description": "Log of optimal temperature for COTS (\u00c2\u00b0C)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Ideal sea surface temperature for crown-of-thorns starfish proliferation",
                "citations": [
                    "https://pubmed.ncbi.nlm.nih.gov/24615941/",
                    "https://link.springer.com/article/10.1007/s00227-022-04027-w",
                    "https://www.aims.gov.au/sites/default/files/cots-revised.pdf"
                ],
                "processed": true
            },
            {
                "parameter": "log_sigma_T",
                "value": 0.6931,
                "description": "Log of temperature tolerance width (\u00c2\u00b0C)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Crown-of-thorns starfish thermal tolerance for survival and feeding activity",
                "citations": [
                    "https://doi.org/10.1007/s00227-014-2441-7",
                    "https://doi.org/10.1016/j.marenvres.2017.01.009"
                ],
                "processed": true
            },
            {
                "parameter": "log_obs_sd",
                "value": -1.6094,
                "description": "Log of observation error standard deviation",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Statistical measurement uncertainty and variability in ecological observations",
                "processed": true
            },
            {
                "parameter": "log_beta",
                "value": -0.223,
                "description": "COTS reproduction efficiency from coral consumption",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Conversion rate of consumed coral tissue into COTS population growth",
                "citations": [
                    "https://doi.org/10.1007/s00338-014-1165-x",
                    "https://doi.org/10.1016/j.ecolmodel.2015.07.001"
                ],
                "processed": true
            }
        ]
    }

## claude 3 7 sonnet Model

This model achieved an objective value of 0.5808.

### Model Intention

    # Ecological Model Improvement: Coral-COTS Feedback Mechanism

    ## Current Model Performance
    The current model shows reasonable tracking of general trends but fails to capture important dynamics:
    1. For COTS populations (cots_pred), the model underestimates major outbreak peaks around 1990-1995
    2. For slow-growing coral (slow_pred), the model significantly underestimates the peak around 1990
    3. For fast-growing coral (fast_pred), the model shows better fit but still misses some fluctuations

    ## Ecological Improvement: Temperature-Enhanced COTS Reproduction
    I've implemented a temperature-dependent reproduction mechanism for COTS. Research shows that warmer temperatures can enhance COTS larval development and survival, leading to population outbreaks. This is particularly important as climate change continues to affect coral reef ecosystems.

    This creates an important ecological feedback:
    - Higher SST $\rightarrow$ Enhanced COTS reproduction $\rightarrow$ Increased coral predation
    - This positive feedback can explain the rapid COTS outbreaks observed in the historical data

    ## Implementation Details
    1. Added a temperature-dependent reproduction modifier for COTS that increases reproduction at higher temperatures
    2. Added a new parameter `temp_repro_threshold` representing the temperature above which COTS reproduction is enhanced
    3. Added a new parameter `temp_repro_effect` controlling the strength of temperature enhancement on reproduction

    This mechanism is ecologically justified because:
    - COTS larval development is temperature-sensitive, with faster development at higher temperatures
    - Warmer waters can increase phytoplankton availability, which is food for COTS larvae
    - Historical COTS outbreaks have been associated with warmer periods
    - The mechanism creates a realistic positive feedback loop in the ecosystem

    The improvement maintains model parsimony while adding an ecologically meaningful mechanism that should better capture the observed dynamics. I've simplified the implementation to use a linear response above the threshold temperature rather than a sigmoid function to improve numerical stability.

### Model Implementation

    #include <TMB.hpp>

    template<class Type>
    Type objective_function<Type>::operator() ()
    {
      // DATA
      DATA_VECTOR(Year);                  // Years of observation
      DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
      DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m^2/year)
      DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
      DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
      DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
      
      // PARAMETERS
      PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
      PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
      PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
      PARAMETER(a_fast);                  // Attack rate on fast-growing coral (m^2/individual/year)
      PARAMETER(a_slow);                  // Attack rate on slow-growing coral (m^2/individual/year)
      PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
      PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
      PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
      PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
      PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
      PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
      PARAMETER(alpha_fs);                // Competition coefficient: effect of slow on fast coral
      PARAMETER(alpha_sf);                // Competition coefficient: effect of fast on slow coral
      PARAMETER(temp_opt);                // Optimal temperature for coral growth (°C)
      PARAMETER(temp_tol);                // Temperature tolerance range (°C)
      PARAMETER(imm_effect);              // Effect of immigration on COTS population
      PARAMETER(coral_threshold);         // Coral cover threshold for COTS survival (% cover)
      PARAMETER(temp_repro_threshold);    // Temperature threshold for enhanced COTS reproduction (°C)
      PARAMETER(temp_repro_effect);       // Effect of temperature on COTS reproduction (dimensionless)
      PARAMETER(sigma_cots);              // Observation error SD for COTS (log scale)
      PARAMETER(sigma_slow);              // Observation error SD for slow-growing coral (log scale)
      PARAMETER(sigma_fast);              // Observation error SD for fast-growing coral (log scale)
      
      // Initialize negative log-likelihood
      Type nll = 0.0;
      
      // Small constant to prevent division by zero
      Type eps = Type(1e-8);
      
      // Number of time steps
      int n_steps = Year.size();
      
      // Vectors to store model predictions
      vector<Type> cots_pred(n_steps);
      vector<Type> slow_pred(n_steps);
      vector<Type> fast_pred(n_steps);
      
      // Initialize with first observation
      cots_pred(0) = cots_dat(0);
      slow_pred(0) = slow_dat(0);
      fast_pred(0) = fast_dat(0);
      
      // Time series simulation
      for (int t = 1; t < n_steps; t++) {
        // Temperature effect on coral growth (Gaussian response curve)
        Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_tol, 2));
        
        // Total coral cover (food availability for COTS)
        Type total_coral = slow_pred(t-1) + fast_pred(t-1);
        
        // Functional responses for COTS feeding on corals (Type II)
        Type denom = 1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1);
        Type F_fast = (a_fast * fast_pred(t-1)) / denom;
        Type F_slow = (a_slow * slow_pred(t-1)) / denom;
        
        // Food limitation effect on COTS (smooth transition at threshold)
        Type food_limitation = 0.1 + 0.9 / (1.0 + exp(-5.0 * (total_coral - coral_threshold)));
        
        // Temperature effect on COTS reproduction
        Type temp_effect_cots = 1.0;
        if (sst_dat(t-1) > temp_repro_threshold) {
          temp_effect_cots = 1.0 + temp_repro_effect * (sst_dat(t-1) - temp_repro_threshold) / 2.0;
        }
        
        // COTS population dynamics
        Type density_factor = std::max(Type(0.0), Type(1.0 - cots_pred(t-1) / K_cots));
        Type cots_growth = r_cots * cots_pred(t-1) * density_factor * food_limitation * temp_effect_cots;
        Type cots_mortality = m_cots * cots_pred(t-1);
        Type cots_immigration = imm_effect * cotsimm_dat(t-1);
        
        cots_pred(t) = std::max(eps, cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration);
        
        // Fast-growing coral dynamics
        Type competition_fast = (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast;
        competition_fast = std::min(Type(1.0), competition_fast);
        
        Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - competition_fast) * temp_effect;
        Type fast_predation = std::min(fast_pred(t-1), F_fast * cots_pred(t-1));
        
        fast_pred(t) = std::max(eps, fast_pred(t-1) + fast_growth - fast_predation);
        
        // Slow-growing coral dynamics
        Type competition_slow = (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow;
        competition_slow = std::min(Type(1.0), competition_slow);
        
        Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - competition_slow) * temp_effect;
        Type slow_predation = std::min(slow_pred(t-1), F_slow * cots_pred(t-1));
        
        slow_pred(t) = std::max(eps, slow_pred(t-1) + slow_growth - slow_predation);
      }
      
      // Calculate negative log-likelihood
      Type min_sigma = Type(0.01);
      
      for (int t = 0; t < n_steps; t++) {
        // COTS abundance likelihood
        Type sigma_cots_t = std::max(min_sigma, sigma_cots);
        nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_t, true);
        
        // Slow-growing coral cover likelihood
        Type sigma_slow_t = std::max(min_sigma, sigma_slow);
        nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_t, true);
        
        // Fast-growing coral cover likelihood
        Type sigma_fast_t = std::max(min_sigma, sigma_fast);
        nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_t, true);
      }
      
      // Report predictions
      REPORT(cots_pred);
      REPORT(slow_pred);
      REPORT(fast_pred);
      
      return nll;
    }

### Model Parameters

    {
        "parameters": [
            {
                "parameter": "r_cots",
                "value": 0.8,
                "description": "Intrinsic growth rate of COTS population (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown-of-thorns starfish population intrinsic reproduction potential",
                "citations": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9023020/",
                    "https://www.sciencedirect.com/science/article/abs/pii/S0025326X18304776",
                    "https://www.mdpi.com/journal/diversity/special_issues/crown_thorns_starfish"
                ],
                "processed": true
            },
            {
                "parameter": "K_cots",
                "value": 3.0,
                "description": "Carrying capacity of COTS population (individuals/m^2)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum sustainable Crown-of-thorns starfish population density",
                "citations": [
                    "https://www.nature.com/articles/s41598-023-46749-x",
                    "https://coralreefs.org/wp-content/uploads/2019/01/Session-72-5-13th-ICRS-Thummasan-et-al-session-72NS.pdf",
                    "https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecs2.4580"
                ],
                "processed": true
            },
            {
                "parameter": "m_cots",
                "value": 0.3,
                "description": "Natural mortality rate of COTS (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Natural mortality rate of Crown-of-thorns starfish population",
                "citations": [
                    "https://www.sciencedirect.com/science/article/pii/S0048969724054329",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9023020/",
                    "https://www.researchgate.net/publication/324765841_Mortality_rates_of_small_juvenile_crown-of-thorns_starfish_Acanthaster_planci_on_the_Great_Barrier_Reef_Implications_for_population_size_and_larval_settlement_thresholds_for_outbreaks"
                ],
                "processed": true
            },
            {
                "parameter": "a_fast",
                "value": 0.4,
                "description": "Attack rate of COTS on fast-growing coral (m^2/individual/year)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Predation rate on fast-growing coral by Crown-of-thorns",
                "citations": [
                    "https://www2.gbrmpa.gov.au/our-work/programs-and-projects/crown-thorns-starfish/coral-predator-outbreaks",
                    "https://www.sciencedirect.com/science/article/pii/S0048969724028389",
                    "https://www.aims.gov.au/research-topics/marine-life/crown-thorns-starfish"
                ],
                "processed": true
            },
            {
                "parameter": "a_slow",
                "value": 0.15,
                "description": "Attack rate of COTS on slow-growing coral (m^2/individual/year)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Predation rate on slow-growing coral by Crown-of-thorns",
                "citations": [
                    "https://www.sciencedirect.com/science/article/pii/S0048969724028389",
                    "https://www.reefresilience.org/pdf/COTS_Nov2003.pdf",
                    "https://www.sciencedirect.com/science/article/pii/S0048969724054329"
                ],
                "processed": true
            },
            {
                "parameter": "h_fast",
                "value": 0.1,
                "description": "Handling time for COTS feeding on fast-growing coral (year/% cover)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Feeding handling time for fast-growing coral species",
                "processed": true
            },
            {
                "parameter": "h_slow",
                "value": 0.2,
                "description": "Handling time for COTS feeding on slow-growing coral (year/% cover)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Feeding handling time for slow-growing coral species",
                "processed": true
            },
            {
                "parameter": "r_fast",
                "value": 0.6,
                "description": "Intrinsic growth rate of fast-growing coral (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Growth rate of fast-recovering branching coral species",
                "citations": [
                    "https://www.sciencedirect.com/science/article/pii/S0960982224001519",
                    "https://www.nature.com/articles/s41598-017-03085-1",
                    "https://www.eurekalert.org/news-releases/1036591"
                ],
                "processed": true
            },
            {
                "parameter": "r_slow",
                "value": 0.2,
                "description": "Intrinsic growth rate of slow-growing coral (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Growth rate of slow-recovering massive coral species",
                "citations": [
                    "https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecy.4510",
                    "https://www.soest.hawaii.edu/soestwp/announce/news/surprising-growth-rates-discovered-in-worlds-deepest-photosynthetic-corals/",
                    "https://www.sciencedirect.com/science/article/pii/S0925857418303094"
                ],
                "processed": true
            },
            {
                "parameter": "K_fast",
                "value": 40.0,
                "description": "Carrying capacity of fast-growing coral (% cover)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum cover potential for fast-growing coral species",
                "citations": [
                    "https://www.aims.gov.au/information-centre/news-and-stories/highest-coral-cover-central-northern-reef-36-years",
                    "https://coralreef.noaa.gov/education/coralfacts.html",
                    "https://www.nature.com/articles/s41467-023-37858-2"
                ],
                "processed": true
            },
            {
                "parameter": "K_slow",
                "value": 60.0,
                "description": "Carrying capacity of slow-growing coral (% cover)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Maximum cover potential for slow-growing coral species",
                "citations": [
                    "https://www.sciencedirect.com/science/article/abs/pii/S0925857422002312",
                    "https://coralreef.noaa.gov/education/coralfacts.html",
                    "https://www.sciencedirect.com/science/article/pii/S0925857418303094"
                ],
                "processed": true
            },
            {
                "parameter": "alpha_fs",
                "value": 0.5,
                "description": "Competition coefficient: effect of slow-growing on fast-growing coral (dimensionless)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Interspecific competition impact of slow on fast corals",
                "processed": true
            },
            {
                "parameter": "alpha_sf",
                "value": 0.3,
                "description": "Competition coefficient: effect of fast-growing on slow-growing coral (dimensionless)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Interspecific competition impact of fast on slow corals",
                "processed": true
            },
            {
                "parameter": "temp_opt",
                "value": 27.0,
                "description": "Optimal temperature for coral growth (\u00c2\u00b0C)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Optimal sea temperature for coral ecosystem health",
                "citations": [
                    "https://coral.org/en/coral-reefs-101/what-do-corals-reefs-need-to-survive/",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8917797/",
                    "https://marsh-reef.org/index.php?threads/best-temperature-for-coral-growth.51752/"
                ],
                "processed": true
            },
            {
                "parameter": "temp_tol",
                "value": 2.0,
                "description": "Temperature tolerance range for coral (\u00c2\u00b0C)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Coral thermal tolerance range for ecosystem resilience",
                "citations": [
                    "https://www.nature.com/articles/s41467-023-40601-6",
                    "https://www.nature.com/articles/srep17639",
                    "https://esajournals.onlinelibrary.wiley.com/doi/10.1002/fee.2483"
                ],
                "processed": true
            },
            {
                "parameter": "imm_effect",
                "value": 0.8,
                "description": "Effect of immigration on COTS population (dimensionless)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Immigration influence on Crown-of-thorns starfish population dynamics",
                "processed": true
            },
            {
                "parameter": "coral_threshold",
                "value": 5.0,
                "description": "Coral cover threshold for COTS survival (% cover)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 3,
                "enhanced_semantic_description": "Minimum coral cover required for starfish population survival",
                "processed": true
            },
            {
                "parameter": "temp_repro_threshold",
                "value": 28.0,
                "description": "Temperature threshold for enhanced COTS reproduction (\u00c2\u00b0C)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Temperature above which COTS larval development and survival is enhanced",
                "citations": [
                    "https://www.sciencedirect.com/science/article/abs/pii/S0141113620305328",
                    "https://www.sciencedirect.com/science/article/abs/pii/S0022098121001751",
                    "https://portlandpress.com/emergtoplifesci/article/6/1/67/230852/Crown-of-thorns-starfish-life-history-traits"
                ],
                "processed": true,
                "found_value": 30.0,
                "found_min": 28.0,
                "found_max": 32.0
            },
            {
                "parameter": "temp_repro_effect",
                "value": 1.5,
                "description": "Effect of temperature on COTS reproduction (dimensionless)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 2,
                "enhanced_semantic_description": "Magnitude of temperature enhancement on COTS reproduction rate",
                "citations": [
                    "https://www.sciencedirect.com/science/article/abs/pii/S0141113620305328",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4325318/",
                    "https://archimer.ifremer.fr/doc/00656/76808/93560.pdf"
                ],
                "processed": true,
                "found_value": 3.02,
                "found_min": 1.2,
                "found_max": 4.9
            },
            {
                "parameter": "sigma_cots",
                "value": 0.2,
                "description": "Observation error standard deviation for COTS (log scale)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 4,
                "enhanced_semantic_description": "Measurement uncertainty for Crown-of-thorns starfish population estimates",
                "processed": true
            },
            {
                "parameter": "sigma_slow",
                "value": 0.3,
                "description": "Observation error standard deviation for slow-growing coral (log scale)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 4,
                "enhanced_semantic_description": "Measurement uncertainty for slow-growing coral cover estimates",
                "processed": true
            },
            {
                "parameter": "sigma_fast",
                "value": 0.3,
                "description": "Observation error standard deviation for fast-growing coral (log scale)",
                "source": "initial estimate",
                "import_type": "PARAMETER",
                "priority": 4,
                "enhanced_semantic_description": "Measurement uncertainty for fast-growing coral cover estimates",
                "processed": true
            }
        ]
    }

## gemini 2.0 flash Model

This model achieved an objective value of 0.5216.

### Model Intention

    Model Assessment:
    The current model captures general trends but struggles with the magnitude and timing of COTS outbreaks. Coral predictions also deviate from observed data. The Holling type II functional response oversimplifies COTS predation.

    Proposed Improvement:
    Incorporate a Holling type III functional response for COTS predation on coral. This includes a quadratic term in the denominator, allowing for a slower initial increase in predation rate at low COTS densities, potentially reflecting COTS aggregation behavior and a saturation constant to limit the magnitude of outbreaks.

### Model Implementation

    #include <TMB.hpp>

    template<class Type>
    Type objective_function<Type>::operator() ()
    {
      // ------------------------------------------------------------------------
      // 1. Data and Parameters:
      // ------------------------------------------------------------------------

      // --- Data: ---
      DATA_VECTOR(Year);              // Time variable (year)
      DATA_VECTOR(cots_dat);          // COTS abundance data (individuals/m2)
      DATA_VECTOR(slow_dat);          // Slow-growing coral cover data (%)
      DATA_VECTOR(fast_dat);          // Fast-growing coral cover data (%)
      DATA_VECTOR(sst_dat);           // Sea surface temperature data (Celsius)
      DATA_VECTOR(cotsimm_dat);       // COTS larval immigration rate (individuals/m2/year)

      // --- Parameters: ---
      PARAMETER(log_r_cots);          // Log of intrinsic growth rate of COTS (year^-1)
      PARAMETER(log_K_cots);          // Log of carrying capacity of COTS (individuals/m2)
      PARAMETER(log_m_cots);          // Log of natural mortality rate of COTS (year^-1)
      PARAMETER(log_p_cots);          // Log of predation rate on COTS (year^-1)
      PARAMETER(log_K1_cots);         // Log of half-saturation constant for COTS predation (individuals/m2)
      PARAMETER(log_a_fast);         // Log of attack rate of COTS on fast-growing coral (m2/individual/year)
      PARAMETER(log_a_slow);         // Log of attack rate of COTS on slow-growing coral (m2/individual/year)
      PARAMETER(log_K_fast);         // Log of carrying capacity of fast-growing coral (%)
      PARAMETER(log_K_slow);         // Log of carrying capacity of slow-growing coral (%)
      PARAMETER(log_r_fast);         // Log of growth rate of fast-growing coral (year^-1)
      PARAMETER(log_r_slow);         // Log of growth rate of slow-growing coral (year^-1)
      PARAMETER(log_m_fast);         // Log of mortality rate of fast-growing coral (year^-1)
      PARAMETER(log_m_slow);         // Log of mortality rate of slow-growing coral (year^-1)
      PARAMETER(log_temp_sensitivity_fast); // Log of temperature sensitivity of fast-growing coral (Celsius^-1)
      PARAMETER(log_temp_sensitivity_slow); // Log of temperature sensitivity of slow-growing coral (Celsius^-1)
      PARAMETER(log_sigma_cots);      // Log of standard deviation of COTS observation error
      PARAMETER(log_sigma_slow);      // Log of standard deviation of slow-growing coral observation error
      PARAMETER(log_sigma_fast);      // Log of standard deviation of fast-growing coral observation error
      PARAMETER(log_h_cots);           // Log of handling time for COTS predation (year)

      // --- Transformations: ---
      Type r_cots   = exp(log_r_cots);
      Type K_cots   = exp(log_K_cots);
      Type m_cots   = exp(log_m_cots);
      Type p_cots   = exp(log_p_cots);
      Type K1_cots  = exp(log_K1_cots);
      Type a_fast  = exp(log_a_fast);
      Type a_slow  = exp(log_a_slow);
      Type K_fast   = exp(log_K_fast);
      Type K_slow   = exp(log_K_slow);
      Type r_fast   = exp(log_r_fast);
      Type r_slow   = exp(log_r_slow);
      Type m_fast   = exp(log_m_fast);
      Type m_slow   = exp(log_m_slow);
      Type temp_sensitivity_fast = exp(log_temp_sensitivity_fast);
      Type temp_sensitivity_slow = exp(log_temp_sensitivity_slow);
      Type sigma_cots = exp(log_sigma_cots);
      Type sigma_slow = exp(log_sigma_slow);
      Type sigma_fast = exp(log_sigma_fast);
      Type h_cots    = exp(log_h_cots);

      // --- Objective function: ---
      Type nll = 0.0; // Initialize negative log-likelihood

      // --- Vectors for predictions: ---
      vector<Type> cots_pred(Year.size());
      vector<Type> slow_pred(Year.size());
      vector<Type> fast_pred(Year.size());

      // --- Initial conditions: ---
      cots_pred(0) = cots_dat(0);    // Initial COTS abundance
      slow_pred(0) = slow_dat(0);    // Initial slow-growing coral cover
      fast_pred(0) = fast_dat(0);    // Initial fast-growing coral cover

      // ------------------------------------------------------------------------
      // 2. Model Equations:
      // ------------------------------------------------------------------------

      for(int t=1; t<Year.size(); t++) {
        // 1. COTS Population Dynamics:
        //    Logistic growth with carrying capacity, predation, and larval immigration.
        Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1) / K_cots);
        Type cots_predation = p_cots * cots_pred(t-1) / (K1_cots + cots_pred(t-1)); // Holling type II
        cots_pred(t) = cots_pred(t-1) + cots_growth - m_cots * cots_pred(t-1) - cots_predation + cotsimm_dat(t);

        // 2. Coral Dynamics:
        //    Logistic growth with COTS predation and temperature-dependent mortality.
        //    COTS preferentially eat fast-growing coral.
        // Holling type III functional response:
        Type fast_predation = a_fast * pow(cots_pred(t-1), 2.0) * fast_pred(t-1) / (pow(h_cots, 2.0) + pow(cots_pred(t-1), 2.0));
        Type temp_mortality_fast = temp_sensitivity_fast * sst_dat(t) * fast_pred(t-1);
        fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1) / (K_fast + slow_pred(t-1))) * (fast_pred(t-1) > Type(0.0)) - m_fast * fast_pred(t-1) - fast_predation - temp_mortality_fast;

        Type slow_predation = a_slow * pow(cots_pred(t-1), 2.0) * slow_pred(t-1) / (pow(h_cots, 2.0) + pow(cots_pred(t-1), 2.0));
        Type temp_mortality_slow = temp_sensitivity_slow * sst_dat(t) * slow_pred(t-1);
        slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1) / (K_slow + fast_pred(t-1))) * (slow_pred(t-1) > Type(0.0)) - m_slow * slow_pred(t-1) - slow_predation - temp_mortality_slow;

        // --- Add small constant to prevent negative values ---
        cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8));
        fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));
        slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));
      }

      // ------------------------------------------------------------------------
      // 3. Likelihood Calculation:
      // ------------------------------------------------------------------------

      for(int t=0; t<Year.size(); t++) {
        // --- COTS likelihood: ---
        nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots, true);

        // --- Slow-growing coral likelihood: ---
        nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_slow, true);

        // --- Fast-growing coral likelihood: ---
        nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_fast, true);

        // --- Parameter penalties: ---
        nll += Type(0.01) * pow(log_r_cots - 0.5, 2.0);
        nll += Type(0.01) * pow(log_K_cots - 2.0, 2.0);
        nll += Type(0.01) * pow(log_m_cots + 0.5, 2.0);
        nll += Type(0.01) * pow(log_p_cots + 1.0, 2.0);
        nll += Type(0.01) * pow(log_K1_cots + 0.5, 2.0);
        nll += Type(0.01) * pow(log_a_fast + 2.0, 2.0);
        nll += Type(0.01) * pow(log_a_slow + 3.0, 2.0);
        nll += Type(0.01) * pow(log_K_fast - 3.0, 2.0);
        nll += Type(0.01) * pow(log_K_slow - 3.0, 2.0);
        nll += Type(0.01) * pow(log_r_fast - 0.8, 2.0);
        nll += Type(0.01) * pow(log_r_slow - 0.3, 2.0);
        nll += Type(0.01) * pow(log_m_fast + 0.2, 2.0);
        nll += Type(0.01) * pow(log_m_slow + 0.7, 2.0);
        nll += Type(0.01) * pow(log_temp_sensitivity_fast + 3.0, 2.0);
        nll += Type(0.01) * pow(log_temp_sensitivity_slow + 3.0, 2.0);
        nll += Type(0.01) * pow(log_sigma_cots + 0.5, 2.0);
        nll += Type(0.01) * pow(log_sigma_slow + 0.5, 2.0);
        nll += Type(0.01) * pow(log_sigma_fast + 0.5, 2.0);
      }

      // ------------------------------------------------------------------------
      // 4. Reporting:
      // ------------------------------------------------------------------------

      REPORT(cots_pred);
      REPORT(slow_pred);
      REPORT(fast_pred);

      return nll;
    }

### Model Parameters

    {
        "parameters": [
            {
                "parameter": "log_r_cots",
                "value": 0.5,
                "description": "Log of intrinsic growth rate of COTS (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown of Thorns starfish population intrinsic growth potential",
                "citations": [
                    "https://link.springer.com/article/10.1007/s00338-024-02560-2",
                    "https://www.sciencedirect.com/science/article/pii/S0048969724054329",
                    "https://www.researchgate.net/figure/Intrinsic-growth-rates-following-different-disturbance-types-The-intrinsic-growth-rate-r_fig7_50597254"
                ],
                "processed": true
            },
            {
                "parameter": "log_K_cots",
                "value": 2.0,
                "description": "Log of carrying capacity of COTS (individuals/m2)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Maximum sustainable Crown of Thorns starfish population density",
                "processed": true
            },
            {
                "parameter": "log_m_cots",
                "value": -0.5,
                "description": "Log of natural mortality rate of COTS (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Natural mortality rate of Crown of Thorns starfish population",
                "citations": [
                    "https://www.sciencedirect.com/science/article/pii/S0048969724054329",
                    "https://www.researchgate.net/publication/324765841_Mortality_rates_of_small_juvenile_crown-of-thorns_starfish_Acanthaster_planci_on_the_Great_Barrier_Reef_Implications_for_population_size_and_larval_settlement_thresholds_for_outbreaks",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9023020/"
                ],
                "processed": true,
                "found_value": -2.355,
                "found_min": -2.355,
                "found_max": -2.355
            },
            {
                "parameter": "log_p_cots",
                "value": -1.0,
                "description": "Log of predation rate on COTS (year^-1)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Predation pressure affecting Crown of Thorns starfish survival",
                "processed": true
            },
            {
                "parameter": "log_K1_cots",
                "value": -0.5,
                "description": "Log of half-saturation constant for COTS predation (individuals/m2)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Half-saturation point for predation impact on COTS",
                "processed": true
            },
            {
                "parameter": "log_a_fast",
                "value": -2.0,
                "description": "Log of attack rate of COTS on fast-growing coral (m2/individual/year)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown of Thorns predation rate on fast-growing coral species",
                "citations": [
                    "https://www.aims.gov.au/research-topics/marine-life/crown-thorns-starfish",
                    "https://www.sciencedirect.com/science/article/pii/S0141113624003167",
                    "https://www2.gbrmpa.gov.au/our-work/programs-and-projects/crown-thorns-starfish/coral-predator-outbreaks"
                ],
                "processed": true
            },
            {
                "parameter": "log_a_slow",
                "value": -3.0,
                "description": "Log of attack rate of COTS on slow-growing coral (m2/individual/year)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Crown of Thorns predation rate on slow-growing coral species",
                "citations": [
                    "https://www.sciencedirect.com/science/article/pii/S0048969724028389",
                    "https://www.aims.gov.au/research-topics/marine-life/crown-thorns-starfish",
                    "https://link.springer.com/article/10.1007/s00338-024-02560-2"
                ],
                "processed": true
            },
            {
                "parameter": "log_K_fast",
                "value": 3.0,
                "description": "Log of carrying capacity of fast-growing coral (%)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Maximum sustainable cover for fast-growing coral species",
                "processed": true
            },
            {
                "parameter": "log_K_slow",
                "value": 3.0,
                "description": "Log of carrying capacity of slow-growing coral (%)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Maximum sustainable cover for slow-growing coral species",
                "processed": true
            },
            {
                "parameter": "log_r_fast",
                "value": 0.8,
                "description": "Log of growth rate of fast-growing coral (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Growth potential of fast-recovering coral reef ecosystems",
                "citations": [
                    "https://www.sciencedaily.com/releases/2024/03/240308123248.htm",
                    "https://www.climateaction.org/news/restored-coral-reefs-can-grow-as-fast-as-healthy-reefs-new-research-shows",
                    "https://www.fisheries.noaa.gov/national/habitat-conservation/restoring-coral-reefs"
                ],
                "processed": true
            },
            {
                "parameter": "log_r_slow",
                "value": 0.3,
                "description": "Log of growth rate of slow-growing coral (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Growth potential of slow-recovering coral reef ecosystems",
                "citations": [
                    "https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.4915",
                    "https://www.sciencedirect.com/science/article/pii/S0960982224001519",
                    "https://oceanservice.noaa.gov/ocean/corals/hope-for-corals.html"
                ],
                "processed": true
            },
            {
                "parameter": "log_m_fast",
                "value": -0.2,
                "description": "Log of mortality rate of fast-growing coral (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Mortality rate for fast-growing coral species",
                "citations": [
                    "https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.725778/full",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4145665/",
                    "https://www.reef2reef.com/threads/mortality-rate.902994/"
                ],
                "processed": true,
                "found_value": -1.2,
                "found_min": -2.99,
                "found_max": 0.0
            },
            {
                "parameter": "log_m_slow",
                "value": -0.7,
                "description": "Log of mortality rate of slow-growing coral (year^-1)",
                "source": "literature",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Mortality rate for slow-growing coral species",
                "citations": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4145665/",
                    "https://link.springer.com/article/10.1007/s00338-023-02440-1",
                    "https://www.reef2reef.com/threads/mortality-rate.902994/"
                ],
                "processed": true,
                "found_value": -1.0,
                "found_min": -1.5,
                "found_max": -0.5
            },
            {
                "parameter": "log_sigma_cots",
                "value": -0.5,
                "description": "Log of standard deviation of COTS observation error",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Measurement uncertainty in Crown of Thorns starfish population",
                "processed": true
            },
            {
                "parameter": "log_sigma_slow",
                "value": -0.5,
                "description": "Log of standard deviation of slow-growing coral observation error",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Measurement uncertainty for slow-growing coral cover",
                "processed": true
            },
            {
                "parameter": "log_sigma_fast",
                "value": -0.5,
                "description": "Log of standard deviation of fast-growing coral observation error",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Measurement uncertainty for fast-growing coral cover",
                "processed": true
            },
            {
                "parameter": "log_temp_sensitivity_fast",
                "value": -3.0,
                "description": "Log of temperature sensitivity of fast-growing coral (Celsius^-1)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Temperature impact on fast-growing coral species resilience",
                "processed": true
            },
            {
                "parameter": "log_temp_sensitivity_slow",
                "value": -3.0,
                "description": "Log of temperature sensitivity of slow-growing coral (Celsius^-1)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Temperature impact on slow-growing coral species resilience",
                "processed": true
            },
            {
                "parameter": "Year",
                "value": [
                    1980,
                    1981,
                    1982,
                    1983,
                    1984,
                    1985,
                    1986,
                    1987,
                    1988,
                    1989,
                    1990,
                    1991,
                    1992,
                    1993,
                    1994,
                    1995,
                    1996,
                    1997,
                    1998,
                    1999,
                    2000,
                    2001,
                    2002,
                    2003,
                    2004,
                    2005
                ],
                "description": "Time variable (year)",
                "source": "Data\\timeseries_data_COTS_response.csv",
                "import_type": "DATA_VECTOR",
                "priority": 0,
                "enhanced_semantic_description": "Temporal progression of Great Barrier Reef ecosystem dynamics",
                "processed": true
            },
            {
                "parameter": "cots_dat",
                "value": [
                    0.2615042,
                    0.5498196,
                    0.7268086,
                    0.5522907,
                    0.828121,
                    0.5470078,
                    0.7580244,
                    0.6287678,
                    0.4761596,
                    0.6465779,
                    0.910707,
                    2.151993,
                    0.6446117,
                    1.672348,
                    0.4765907,
                    0.8075009,
                    0.3634731,
                    0.3727647,
                    0.6172546,
                    0.3106559,
                    0.2560048,
                    0.2983628,
                    0.3362447,
                    0.2878112,
                    0.3220782,
                    0.4308113
                ],
                "description": "COTS abundance data (individuals/m2)",
                "source": "Data\\timeseries_data_COTS_response.csv",
                "import_type": "DATA_VECTOR",
                "priority": 0,
                "enhanced_semantic_description": "Crown of Thorns starfish population density measurements",
                "processed": true
            },
            {
                "parameter": "slow_dat",
                "value": [
                    10.069386,
                    17.723573,
                    23.827578,
                    19.418118,
                    15.635417,
                    32.459609,
                    21.556739,
                    34.19385,
                    18.870792,
                    27.317593,
                    44.330127,
                    13.327491,
                    11.37412,
                    8.33619,
                    15.20695,
                    12.443031,
                    11.066606,
                    10.644162,
                    2.362755,
                    2.760007,
                    3.723647,
                    7.035517,
                    10.223949,
                    8.475412,
                    9.081878,
                    13.731449
                ],
                "description": "Slow-growing coral cover data (%)",
                "source": "Data\\timeseries_data_COTS_response.csv",
                "import_type": "DATA_VECTOR",
                "priority": 0,
                "enhanced_semantic_description": "Slow-growing coral species percentage cover measurements",
                "processed": true
            },
            {
                "parameter": "fast_dat",
                "value": [
                    12.772605,
                    16.414745,
                    12.777292,
                    12.279754,
                    15.115161,
                    12.819409,
                    10.463078,
                    9.725137,
                    13.201352,
                    16.4512,
                    11.139259,
                    18.083162,
                    11.522349,
                    8.509987,
                    9.162216,
                    5.335342,
                    8.604409,
                    7.278116,
                    2.629035,
                    4.695132,
                    1.594753,
                    5.217158,
                    2.60407,
                    3.361801,
                    7.328911,
                    4.401384
                ],
                "description": "Fast-growing coral cover data (%)",
                "source": "Data\\timeseries_data_COTS_response.csv",
                "import_type": "DATA_VECTOR",
                "priority": 0,
                "enhanced_semantic_description": "Fast-growing coral species percentage cover measurements",
                "processed": true
            },
            {
                "parameter": "sst_dat",
                "value": [
                    28.1,
                    28.2,
                    29.2,
                    24.9,
                    27.8,
                    28.6,
                    26.9,
                    26.5,
                    26.8,
                    25.2,
                    26.3,
                    25.9,
                    26.9,
                    25.8,
                    23.2,
                    28.9,
                    30.2,
                    31.4,
                    25.4,
                    25.7,
                    24.5,
                    26.1,
                    26.8,
                    27.8,
                    25.1,
                    26.3
                ],
                "description": "Sea surface temperature data (Celsius)",
                "source": "Data\\timeseries_data_COTS_forcing.csv",
                "import_type": "DATA_VECTOR",
                "priority": 0,
                "enhanced_semantic_description": "Sea surface temperature variations affecting reef ecosystem",
                "processed": true
            },
            {
                "parameter": "cotsimm_dat",
                "value": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1.5,
                    0,
                    1.6,
                    0.7,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "description": "COTS larval immigration rate (individuals/m2/year)",
                "source": "Data\\timeseries_data_COTS_forcing.csv",
                "import_type": "DATA_VECTOR",
                "priority": 0,
                "enhanced_semantic_description": "Crown of Thorns starfish larval immigration and dispersal rate",
                "processed": true
            },
            {
                "parameter": "log_h_cots",
                "value": -1.0,
                "description": "Log of handling time for COTS predation (year)",
                "source": "expert opinion",
                "import_type": "PARAMETER",
                "priority": 1,
                "enhanced_semantic_description": "Handling time for Crown of Thorns starfish predation interactions",
                "processed": true
            }
        ]
    }

[^1]: Corresponding author: scott.spillias@csiro.au
