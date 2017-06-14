from ..inspect.plot import normhist
from .cfm import Confusion, ConfusionGrid

def best_params(grid, bestof=20):
    return sorted(grid.grid_scores_,
                  key=lambda tup: tup[1],
                  reverse=True)[:bestof]

# FIXME plotting multiple figures in a single ipython cell is not working yet
def evaluate(grid, clf, X_train, X_test, y_train, y_test):
    score = grid.score(X_test, y_test)
    print("Test Accuracy: {0}\n".format(score))
    print("Grid Score distribution:")
    normhist([x[1] for x in grid.grid_scores_])
    print("Best grid scores:")
    best_params(grid)
    #Confusion(grid, X_test, y_test).plot()
    #ConfusionGrid(grid, clf, X_train, X_test, y_train, y_test).plot()
