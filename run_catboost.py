from catboost import CatBoostClassifier, Pool, metrics, cv
import hyperopt
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from numpy.random import default_rng

from preprocessor import DatasetSplits

default_params = {
    'iterations': 2000,
    'random_seed': 42,
    'eval_metric': metrics.Accuracy(),
    # 'logging_level': 'Silent',
    'use_best_model': True,
    # Early stopping
    'od_type': 'Iter',
    'od_wait': 100,
}

# {'l2_leaf_reg': 7.0, 'learning_rate': 0.4953753923070011}
# {'l2_leaf_reg': 6.0, 'learning_rate': 0.4772753822091944}


class CatBoostRunner:
    def __init__(self, dataset_splits: DatasetSplits):
        self.X_train = dataset_splits.X_train
        self.y_train = dataset_splits.y_train
        self.X_validation = dataset_splits.X_validation
        self.y_validation = dataset_splits.y_validation
        self.X_test = dataset_splits.X_test
        self.y_test = dataset_splits.y_test

        self.train_pool = Pool(self.X_train, self.y_train)
        self.validation_pool = Pool(self.X_validation, self.y_validation)

    def run_catboost(self, additional_params={}):
        params = default_params.copy()
        params.update(additional_params)

        model = CatBoostClassifier(**params)
        model.fit(self.train_pool, **{
                  'eval_set': self.validation_pool,
                  'plot': True
                  })

        feature_importances = model.get_feature_importance(self.train_pool)
        feature_names = self.X_train.columns
        for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
            print('{}: {}'.format(name, score))

        cv_params = model.get_params()
        cv_params.update({
            'loss_function': metrics.Logloss()
        })
        cv_data = cv(self.train_pool, cv_params, plot=True)

        print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
            np.max(cv_data['test-Accuracy-mean']),
            cv_data['test-Accuracy-std'][np.argmax(
                cv_data['test-Accuracy-mean'])],
            np.argmax(cv_data['test-Accuracy-mean'])
        ))
        print('Precise validation accuracy score: {}'.format(
            np.max(cv_data['test-Accuracy-mean'])))

        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(classification_report(self.y_test, y_pred))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()
        plt.show()

    def parameter_tuning(self):
        def hyperopt_objective(params):
            model = CatBoostClassifier(
                l2_leaf_reg=int(params['l2_leaf_reg']),
                learning_rate=params['learning_rate'],
                iterations=500,
                eval_metric=metrics.Accuracy(),
                random_seed=42,
                verbose=False,
                loss_function=metrics.Logloss(),
            )
            cv_params = model.get_params()
            # cv_params.update({
            #     'loss_function': metrics.Logloss()
            # })
            cv_data = cv(self.train_pool, cv_params)

            best_accuracy = np.max(cv_data['test-Accuracy-mean'])

            return 1 - best_accuracy  # as hyperopt minimises

        params_space = {
            'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
        }

        trials = hyperopt.Trials()

        best = hyperopt.fmin(
            hyperopt_objective,
            space=params_space,
            algo=hyperopt.tpe.suggest,
            max_evals=50,
            trials=trials,
            rstate=default_rng(123)
        )

        print('best:', best)
        return best
