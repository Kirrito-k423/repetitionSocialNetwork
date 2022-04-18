from data import DataBase

db = DataBase()
[dataset, edge_index] = db.exampleDataFrom()
print(edge_index.size())
print(dataset.size())
