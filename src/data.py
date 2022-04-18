import torch
import psycopg2
from rich.progress import track
from torch.utils.data import TensorDataset

USER_CONFIG = {
    'user': 'acsacom',
    'password': 'acsacom',
    'host': '127.0.0.1',
    'port': 5432
}


class DataBase(object):
    def __init__(self, user=USER_CONFIG['user'], password=USER_CONFIG['password'], host=USER_CONFIG['host'], port=USER_CONFIG['port']):
        """init database with default config

        Args:
            user (str, optional): user. Defaults to 'acsacom'.
            password (str, optional): password. Defaults to 'acsacom'.
            host (str, optional): host ip. Defaults to '127.0.0.1'.
            port (int, optional): connecting port. Defaults to 5432.
        """
        self.conn = psycopg2.connect(
            database='meetup', user=user, password=password, host=host, port=port)

    def exampleDataFrom(self):
        cursor = self.conn.cursor()

        # get edge
        sql = 'select array_agg(rsvps.memberid) from rsvps group by eventid'
        cursor.execute(sql)
        members_in_same_event = cursor.fetchall()
        members_in_same_event = [_[0] for _ in members_in_same_event]
        members = sorted(self.query('members', ['id']))
        members_with_index = dict(
            zip([_[0] for _ in members], range(len(members))))
        edge_index = [[], []]
        for event_members in track(members_in_same_event, description="start to get edges..."):
            for index_i, member_i in enumerate(event_members):
                for index_j, member_j in enumerate(event_members):
                    if index_i != index_j and int(member_i) in members_with_index and int(member_j) in members_with_index:
                        edge_index[0].append(members_with_index[int(member_i)])
                        edge_index[1].append(members_with_index[int(member_j)])
                        edge_index[0].append(members_with_index[int(member_j)])
                        edge_index[1].append(members_with_index[int(member_i)])
        edge_index_ = torch.tensor(edge_index, dtype=torch.long)

        # get tranGroup features
        group_ids = [_[0] for _ in sorted(self.query('groups', ['id']))]
        topic_ids = [_[0] for _ in sorted(self.query('topics', ['id']))]
        group_topics = sorted(self.query('grouptopics'))
        groups_with_index = dict(zip(group_ids, range(len(group_ids))))
        topics_with_index = dict(zip(topic_ids, range(len(topic_ids))))
        x = [[0 for __ in range(len(topic_ids))]
             for _ in range(len(group_ids))]
        for group, topic in track(group_topics, description='start to get train group features...'):
            # print(f'group {group} has topic {topic}')
            x[groups_with_index[group]][topics_with_index[topic]] = 1
        x_ = torch.tensor(x, dtype=torch.float,)

        # get label
        label = [[0 for __ in members] for _ in group_ids]
        sql = 'select memberid, array_agg(membergroups.groupid) from membergroups group by memberid'
        cursor.execute(sql)
        membergroups = cursor.fetchall()
        for memberid, groupids in track(membergroups, description="start to get label..."):
            for groupid in groupids:
                if memberid in members_with_index and groupid in groups_with_index:
                    label[groups_with_index[groupid]
                          ][members_with_index[memberid]] = 1
        label_ = torch.tensor(label, dtype=torch.float)
        dataset = TensorDataset(x_, label_)
        print(x_.size())
        print(label_.size())
        print(edge_index_.size())
        groupNum=x_.size()[0]
        topicNum=x_.size()[1]
        nodeNum=label_.size()[1]
        edgeNum=edge_index_.size()[1]

        return [dataset, edge_index_,nodeNum,edgeNum,topicNum,groupNum]

    def query(self, tablename: str, attrs=[], conditions=[]) -> list:
        """query from meetup data, returning results as a list.
        Remember to add \' when query string

        Args:
            tablename (str): table name
            attrs (list): attrbuites list
            conditions (list): a list of conditions
        """
        for attr in attrs:
            if type(attr) is not str:
                raise TypeError(
                    f"Every attributes must be a string, {attr} is {type(attr)}")
        attr_str = ', '.join(attrs) if attrs else ' * '
        condition_str = 'where ' if conditions and 'order by' not in conditions[
            0] and 'limit' not in conditions[0] else ''
        for index, condition in enumerate(conditions):
            if ('order by' not in condition and 'limit' not in condition) \
                    and (index < len(conditions) - 1) \
                    and ('order by' not in conditions[index+1] and 'limit' not in conditions[index+1]):
                condition_str += f' {condition} and '
            elif ('order by' not in condition and 'limit' not in condition) \
                    and (index < len(conditions) - 1) \
                    and ('order by' in conditions[index+1] or 'limit' in conditions[index+1]):
                condition_str += f' {condition} '
            elif ('order by' in condition or 'limit' in condition) \
                    or (index == len(conditions) - 1):
                condition_str += f' {condition} '
            else:
                raise ValueError(
                    f"case should not achieved, input error, index: {index}, condition: {condition}")
        sql = f'select {attr_str} from {tablename} {condition_str}'
        cursor = self.conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        result = [[_[i] for i in range(len(_))] for _ in result]
        print(f"sql: {sql}")
        return result

    def __del__(self):
        self.conn.close()


if __name__ == '__main__':
    db = DataBase()
    db.exampleDataFrom()
