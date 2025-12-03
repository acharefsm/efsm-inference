import csv

import deap_gp
import efsm
import gp_fitness
import networkx as nx
import numpy as np
from generalise_helper import efsm_to_dot, efsm_to_json
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted


def infer_output(
    samples, mu_size, lambda_size, generation_size, mut_proba, max_init_depth, max_depth, fitness_type, **kwargs
):
    global total_correct

    pset = deap_gp.setup_pset(samples)
    best = deap_gp.run_gp(
        mut_proba,
        samples,
        pset,
        mu=mu_size,
        lamb=lambda_size,
        max_init=max_init_depth,
        max_depth=max_depth,
        ngen=generation_size,
        type_=fitness_type,
        **kwargs,
    )

    args = samples[samples.columns[:-1]]
    outputs = samples[samples.columns[-1]]

    correct = gp_fitness.correct(best, samples, pset, [() for i in range(len(samples))])

    if not correct:
        total_correct += 1
        bf = deap_gp.gp.compile(expr=best, pset=pset)
        predicted = args.apply(lambda args: bf(**(args.to_dict())), axis=1)
        correct = outputs == predicted
        # print("guard inferred ",str(best))
        # print("samples",samples)
        # print("correct",correct)

    return str(best)


def get_total_correct():
    global total_correct

    return total_correct


class EFSMGeneraliserEstimator(BaseEstimator):
    def __init__(
        self,
        # hyperparameters to search (defaults are the first elements from lists)
        mu_size=20,
        lambda_size=5,
        generation_size=25,
        mutation_prob=0.1,
        max_init_depth=1,
        max_depth=5,
        fitness_type="step",
        conjecture_efsm=None,
        original=None,
        conjecture_path=None,
        random_seed=None,
        infer_output=None,
        total_wrong_fn=None,
    ):
        self.mu_size = mu_size
        self.lambda_size = lambda_size
        self.generation_size = generation_size
        self.mutation_prob = mutation_prob
        self.max_init_depth = max_init_depth
        self.max_depth = max_depth
        self.fitness_type = fitness_type

        # dependencies injected once (not hyperparameters)
        self.conjecture_efsm = conjecture_efsm
        self.original = original
        self.conjecture_path = conjecture_path
        self.random_seed = random_seed

        self.infer_output = infer_output
        self.total_wrong_fn = total_wrong_fn

    def fit(self, X=None, y=None):
        """
        Runs efsm.generalise with the current hyperparameters and computes metrics.
        We store `total_wrong` on the fitted estimator for later use.
        """
        # sanity checks
        if self.conjecture_efsm is None:
            raise ValueError("sampled efsm (the EFSM model to generalise) must be provided")
        if self.original is None:
            raise ValueError("original (the original EFSM graph) must be provided")

        global total_correct

        total_correct = 0

        generalised = efsm.generalise(
            self.mu_size,
            self.lambda_size,
            self.generation_size,
            self.mutation_prob,
            self.max_init_depth,
            self.max_depth,
            self.fitness_type,
            self.conjecture_efsm,
            self.infer_output,
            random_seed=self.random_seed,
        )

        if self.conjecture_path:
            dot_path = self.conjecture_path.replace(
                ".dot",
                f"_generalised_{self.mu_size}_{self.lambda_size}_{self.generation_size}_"
                f"{self.mutation_prob}_{self.max_init_depth}_{self.max_depth}_{self.fitness_type}.dot",
            )
            json_path = dot_path.replace(".dot", ".json")
            try:
                efsm_to_dot(generalised, dot_path)
                efsm_to_json(generalised, json_path)
            except Exception as e:
                print(e)
                raise ValueError("conjecture path must be valid")

        if callable(self.total_wrong_fn):
            try:
                total_wrong_val = self.total_wrong_fn()
            except Exception as e:
                print(e)
                total_wrong_val = None
        else:
            raise ValueError("Total wrong function not avaiable")

        # store attributes
        self.generalised = generalised
        self.total_wrong = total_wrong_val if total_wrong_val is not None else float("inf")

        return self

    def score(self, X=None, y=None):
        """
        GridSearchCV maximizes the score. We want to minimize total_wrong, so return -total_wrong.
        """
        check_is_fitted(self, "total_wrong")
        return -float(self.total_wrong)


def run_experiment(trace, conjecture_path, seed, n_jobs):
    original = nx.nx_pydot.read_dot(conjecture_path)
    conjecture_efsm = efsm.efsm(original)

    # param_grid = {
    #     "mu_size": [20, 40, 60, 80, 100],
    #     "lambda_size": [5, 10, 15, 20, 25],
    #     "generation_size": [25, 50, 75, 100, 125],
    #     "mutation_prob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     "max_init_depth": [1, 3, 7],
    #     "max_depth": [3, 5, 7, 15],
    #     "fitness_type": ["step", "continuous", "recursive"]
    # }

    param_grid = {
        "mu_size": [50],
        "lambda_size": [5],
        "generation_size": [50],
        "mutation_prob": [0.5],
        "max_init_depth": [1],
        "max_depth": [5],
        "fitness_type": ["step"],
    }

    base_est = EFSMGeneraliserEstimator(
        conjecture_efsm=conjecture_efsm,
        original=original,
        conjecture_path=conjecture_path,
        random_seed=seed,
        infer_output=infer_output,
        total_wrong_fn=get_total_correct,
    )

    X = np.zeros((1, 1))
    y = np.zeros((1,))

    cv_splits = [(np.array([0]), np.array([0]))]  # single split; fit() ignores X/y

    gs = GridSearchCV(
        estimator=base_est,
        param_grid=param_grid,
        scoring=None,  # estimator.score will be used (i.e. -total_wrong)
        cv=cv_splits,
        refit=False,
        n_jobs=n_jobs,
        verbose=2,
    )

    gs.fit(X, y)

    headers = [
        "mu_size",
        "lambda_size",
        "generation_size",
        "mutation_prob",
        "max_init_depth",
        "max_depth",
        "fitness_type",
        "total_wrong",
        "mean_test_score",
    ]

    with open("experiment_results___.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # results_ contains one row per parameter combination
        results = gs.cv_results_
        params_list = results["params"]
        mean_test_scores = results[
            "mean_test_score"
        ]  # this is -total_wrong (since estimator.score returns -total_wrong)

        for params, mean_score in zip(params_list, mean_test_scores):
            # mean_score is negative total_wrong (or NaN if fit failed)
            est = EFSMGeneraliserEstimator(
                conjecture_efsm=conjecture_efsm,
                original=original,
                conjecture_path=conjecture_path,
                infer_output=None,
                random_seed=seed,
                total_wrong_fn=None,
            )

            est.set_params(**params)
            try:
                est.fit(X=None, y=None)
                total_wrong_val = est.total_wrong
            except Exception:
                levenshtein_val = None
                total_wrong_val = None

            writer.writerow(
                [
                    params["mu_size"],
                    params["lambda_size"],
                    params["generation_size"],
                    params["mutation_prob"],
                    params["max_init_depth"],
                    params["max_depth"],
                    params["fitness_type"],
                    total_wrong_val,
                    mean_score,
                ]
            )

    try:
        best_idx = np.nanargmax(results["mean_test_score"])
        best_params = results["params"][best_idx]
        best_score = results["mean_test_score"][best_idx]
        print("Best params (min total_wrong):", best_params)
        print("Best mean_test_score (=-total_wrong):", best_score)
    except Exception:
        print("GridSearch finished. Inspect experiment_results.csv for per-configuration metrics.")

    # data_s1_s0 = [
    #     [1234, 1000, 2, 2345, True],
    #     [2345, -500, 2, -9999, True],
    #     [2345, -500, 2, 1234, True],
    #     [1234, 1000, 0, -9999, False],
    #     [1234, 1000, 0, 2345, False],
    #     [1234, 1000, 1, -9999, False],
    #     [2345, -500, 0, 1234, False],
    #     [2345, -500, 1, 1234, False],
    #     [1234, 1000, 1, 2345, False],
    #     [2345, -500, 1, -9999, False],
    #     [2345, -500, 0, -9999, False],
    #     [1234, 1000, 1, 1234, False],
    #     [1234, 1000, 0, 1234, False],
    #     [2345, -500, 0, 2345, False],
    #     [2345, -500, 2, 2345, False],
    #     [2345, -500, 1, 2345, False],
    #     [1234, 1000, 2, 1234, False]
    # ]
    # df_s1_s0 = pd.DataFrame(data_s1_s0, columns=["r0", "r1", "r2", "i0", "guard"])
    # df_s1_s0["target"] = "s1->s0"

    # data_s1_s2 = [
    #     [1234, 1000, 2, 2345, False],
    #     [2345, -500, 2, -9999, False],
    #     [2345, -500, 2, 1234, False],
    #     [1234, 1000, 0, -9999, False],
    #     [1234, 1000, 0, 2345, False],
    #     [1234, 1000, 1, -9999, False],
    #     [2345, -500, 0, 1234, False],
    #     [2345, -500, 1, 1234, False],
    #     [1234, 1000, 1, 2345, False],
    #     [2345, -500, 1, -9999, False],
    #     [2345, -500, 0, -9999, False],
    #     [1234, 1000, 1, 1234, True],
    #     [1234, 1000, 0, 1234, True],
    #     [2345, -500, 0, 2345, True],
    #     [2345, -500, 2, 2345, True],
    #     [2345, -500, 1, 2345, True],
    #     [1234, 1000, 2, 1234, True]
    # ]
    # df_s1_s2 = pd.DataFrame(data_s1_s2, columns=["r0", "r1", "r2", "i0", "guard"])
    # df_s1_s2["target"] = "s1->s2"

    # data_s1_s1 = [
    #     [1234, 1000, 2, 2345, False],
    #     [2345, -500, 2, -9999, False],
    #     [2345, -500, 2, 1234, False],
    #     [1234, 1000, 0, -9999, True],
    #     [1234, 1000, 0, 2345, True],
    #     [1234, 1000, 1, -9999, True],
    #     [2345, -500, 0, 1234, True],
    #     [2345, -500, 1, 1234, True],
    #     [1234, 1000, 1, 2345, True],
    #     [2345, -500, 1, -9999, True],
    #     [2345, -500, 0, -9999, True],
    #     [1234, 1000, 1, 1234, False],
    #     [1234, 1000, 0, 1234, False],
    #     [2345, -500, 0, 2345, False],
    #     [2345, -500, 2, 2345, False],
    #     [2345, -500, 1, 2345, False],
    #     [1234, 1000, 2, 1234, False]
    # ]
    # df_s1_s1 = pd.DataFrame(data_s1_s1, columns=["r0", "r1", "r2", "i0", "guard"])
    # df_s1_s1["target"] = "s1->s1"

    # df_all = pd.concat([df_s1_s0, df_s1_s1, df_s1_s2])

    # train_data = df_all[df_all["guard"] == True].copy()

    # for df in [df_all, train_data]:
    #     df["i0_eq_r0"] = (df["i0"] == df["r0"]).astype(int)
    #     df["r2_ge_2"] = (df["r2"] >= 2).astype(int)
    #     df["i0_is_neg"] = (df["i0"] < 0).astype(int)

    # X = train_data[["i0_eq_r0", "r2_ge_2", "i0_is_neg"]]
    # y = train_data["target"]

    # clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    # clf.fit(X, y)

    # print("\nDecision Tree Rules:")
    # print(export_text(clf, feature_names=list(X.columns)))

    # df_all["predicted_transition"] = clf.predict(df_all[["i0_eq_r0", "r2_ge_2", "i0_is_neg"]])

    # print("\nPredicted transitions:")
    # print(df_all[["r0", "r1", "r2", "i0", "guard", "target", "predicted_transition"]])

    # plt.figure(figsize=(10,6))
    # plot_tree(clf, feature_names=list(X.columns), class_names=clf.classes_, filled=True, rounded=True)
    # plt.show()
