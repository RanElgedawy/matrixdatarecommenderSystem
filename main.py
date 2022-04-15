
from surprise import Dataset, SVD
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import KNNWithMeans
from pathlib import Path


ratingsPath= Path("ratings_small.csv")
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(ratingsPath, reader=reader)

#Probabilistic Matrix Factorizarion
PMF = SVD()
cross_validate(PMF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#Item based collaborative filtering
sim_options = {
    "name": "msd",
    "user_based": False,  # Compute  similarities between items
}
itemCF = KNNWithMeans(k=100, sim_options=sim_options)

cross_validate(itemCF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#User based collaborative filtering
sim_options1 = {
    "name": "msd",
    "user_based": True,  # Compute  similarities between items
}
userCF = KNNWithMeans(k=100, sim_options=sim_options1)

cross_validate(userCF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



