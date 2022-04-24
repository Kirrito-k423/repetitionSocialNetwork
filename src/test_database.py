import unittest
import pprint
from database import DataBase


class TestDataBase(unittest.TestCase):
    def test_query(self):
        db = DataBase()
        test_case = [
            {'tablename': 'groups', 'attrs': ['id', 'name'], 'conditions':[
                'rating > 4.3', 'members > 100', 'limit 10']},
            {'tablename': 'groups', 'attrs': [],
                'conditions': ['urlname = \'science-88\'', 'limit 20']},
            {'tablename': 'groups', 'attrs': [], 'conditions': [
                'join_mode = \'open\'', 'lat > 42', 'order by id', 'limit 10']}
        ]
        for index, case in enumerate(test_case):
            print(f'\n=======   Test case {index}  start! ==========\n')
            result = db.query(case['tablename'],
                              case['attrs'], case['conditions'])
            pprint.pprint(result)
            print(f'\n=======   Test case {index}   fininsh! ==========\n')


if __name__ == '__main__':
    unittest.main()
