# Human-Decision-Making-Under-Uncertainty

*   Στοιχείο λίστας
*   Στοιχείο λίστας



The [choices13k](https://github.com/jcpeterson/choices13k) contains
human decision rates on 13,006 risky choice problems. The purpose of
the dataset is to study how people make decisions under uncertaintly.

Participants in the study were presented with two decision scenarios
("gambles") and were asked to select the most appealing of them. Both
gambles were represented to participants as a list of rewards and
their associated probabilities. Based on that, the participants
selected either Gamble A or Gamble B.

Your objective in this assignment is to predict, as best as you can,
the frequency with which the participants selected Gamble B (column
`bRate` in the dataset).

You will find the description of the dataset in its repo, as well as a
small notebook example.

With thanks to the researchers who made this dataset:

* Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., &
  Griffiths, T. L. (2021). Using large-scale experiments and machine
  learning to discover theories of human decision-making.
  *Science* 372:1209-1214.

* Bourgin, D. D., Peterson, J. C., Reichman, D., Russell, S. J., &
  Griffiths, T. L. (2019). Cognitive model priors for predicting human
  decisions. In *Proceedings of the 36th International Conference on
  Machine Learning (PMLR)*, 97:5133-5141.
  ## Requirements

* You may use `c13k_selections.csv` and `c13_problems.json`.

* You can use the existing features or add additional, derived
  features from the dataset. Explain the rationale for the inclusion
  of each feature (either the existing one, or additional one) in your analysis.

* You must use at least three different machine learning methods and
  one neural network method.

* You must perform appropriate hyperparameter tuning.

* After identifying the best approach, you must document which
  features are important for the prediction, and how.

* You must do appropriate train-test splitting and cross-validation as
  appropriate.

* You will report the following metrics for each approach:

  * MAE (primary)
  * $R^2$
  * RMSE

* In the very end of your notebook submission you will add a cell with
  the best MAE you achieved (properly cross-validated).
  ## Evaluation

Your work will be evaluated for completeness and quality. There is no absolute measure by which a 10/10 is awarded. The best submissions will get the top grades, and the rest will be graded accordingly. Grading will take into account:

* Is the work engaging?

* Is the approach explained well enough?

* Have the questions been answered perfunctorily, or do the answers exhibit attention and care?
