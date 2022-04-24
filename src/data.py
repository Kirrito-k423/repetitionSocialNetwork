from sklearn.semi_supervised import LabelSpreading
import os
import json
import torch
import psycopg2
import pandas as pd
from datetime import datetime
from copy import deepcopy
from rich.progress import track, Progress
from torch.utils.data import TensorDataset

USER_CONFIG = {
    "user": "acsacom",
    "password": "acsacom",
    "host": "127.0.0.1",
    "port": 5432,
}

DATA_DIR = "/home/acsacom/Meetup/data/data"
FILE_CONFIG = {
    "members": os.path.join(DATA_DIR, "member.csv"),
    "events": os.path.join(DATA_DIR, "event.csv"),
    "topics": os.path.join(DATA_DIR, "topics.csv"),
    "rsvp": os.path.join(DATA_DIR, "rsvp.csv"),
    "groups": os.path.join(DATA_DIR, "group.csv"),
    "member_groups": os.path.join(DATA_DIR, "member_group.json"),
}


class DataBase(object):
    def __init__(
        self,
        user=USER_CONFIG["user"],
        password=USER_CONFIG["password"],
        host=USER_CONFIG["host"],
        port=USER_CONFIG["port"],
    ):
        """init database with default config

        Args:
            user (str, optional): user. Defaults to 'acsacom'.
            password (str, optional): password. Defaults to 'acsacom'.
            host (str, optional): host ip. Defaults to '127.0.0.1'.
            port (int, optional): connecting port. Defaults to 5432.
        """
        self.conn = psycopg2.connect(
            database="meetup", user=user, password=password, host=host, port=port
        )
        self.simple_topics = False
        self.membernum = 0
        self.memberids = []
        self.groupids = []
        self.topicids = []
        self.group_topics = []
        self.member_index = dict()  # key: memberid, value: member index in dataset
        self.group_index = dict()  # key: groupid, value: group index in dataset
        # select a representative topic from a collections of similarity topics
        self.topic_mapping = dict()  # key: topicid, value: representative topicid
        # [{representative topic name}, [topic_ids], [topic_names]]
        self.topic_info = dict()
        self.topic_index = dict()  # key: topicid, value: topics(id) index in dataset
        self.member_groups = dict()  # key: memberid, value: groups(id) a member joined
        self.event_members = (
            dict()
        )  # key: eventid, value: members(id) participated in event
        self.edge_index = [[]]
        # self.membergroups = [[]]
        self.labels = [[]]
        self.group_features = []

    def init_data(self):
        self.init_events()
        self.init_groups()
        self.init_member()
        self.init_rsvp()
        self.init_topics()
        self.init_member_groups()

    def init_member(self):
        # Init members table from csv file. Attention: replace '' with '!
        member_file = FILE_CONFIG["members"]
        data = pd.read_csv(
            member_file, skipinitialspace=True, escapechar="\\", quotechar='"'
        )
        data["id"].astype("int")
        data["lat"].astype("float")
        data["lon"].astype("float")
        data["joined"] = data["joined"].astype("int")
        data["visited"].astype("int")
        cur = self.conn.cursor()
        for row in track(
            data.iterrows(), description="inserting data", total=len(data)
        ):
            c = row[1]
            sql = (
                "insert into members(id, name, link, lon, lat, country, city, hometown, joined, lang, visited)"
                " values ({}, '{}', '{}', {}, {}, '{}', '{}', '{}', '{}', '{}','{}')"
            ).format(
                c.id,
                c.test_name,
                c.link,
                c.lon,
                c.lat,
                c.country,
                c.city,
                c.hometown,
                datetime.strftime(
                    datetime.fromtimestamp(c.joined / 1000), "%Y-%m-%d_%H:%M:%S"
                ),
                c.lang,
                datetime.strftime(
                    datetime.fromtimestamp(c.visited / 1000), "%Y-%m-%d_%H:%M:%S"
                ),
            )
            cur.execute(sql)
        self.conn.commit()

    def init_events(self):
        filepath = FILE_CONFIG["events"]
        data = pd.read_csv(
            filepath, skipinitialspace=True, escapechar="\\", quotechar='"'
        )
        data["id"].astype("str")
        data["joined"] = data["joined"].astype("int")
        data["visited"].astype("int")
        data["groupid"].astype("int")
        cur = self.conn.cursor()
        for row in track(
            data.iterrows(), description="inserting event...", total=len(data)
        ):
            c = row[1]
            lat = 0.0 if c.lat == "virtual" else float(c.lat)
            lon = 0.0 if c.lon == "virtual" else float(c.lon)
            sql = (
                "insert into events(id, name, event_url, lon, lat, city, groupid, created, visibility, time)"
                " values ('{}', '{}', '{}', {}, {}, '{}', {}, '{}', '{}', '{}')"
            ).format(
                c.id,
                c.test_name,
                c.event_url,
                lon,
                lat,
                c.city,
                c.groupid,
                datetime.strftime(
                    datetime.fromtimestamp(c.created / 1000), "%Y-%m-%d_%H:%M:%S"
                ),
                c.visibility,
                datetime.strftime(
                    datetime.fromtimestamp(c.time / 1000), "%Y-%m-%d_%H:%M:%S"
                ),
            )
            cur.execute(sql)
        self.conn.commit()

    def init_topics(self):
        filepath = FILE_CONFIG["topics"]
        data = pd.read_csv(
            filepath, skipinitialspace=True, escapechar="\\", quotechar='"'
        )
        data["id"].astype("int")
        cur = self.conn.cursor()
        for row in track(
            data.iterrows(), description="inserting topics...", total=len(data)
        ):
            c = row[1]
            sql = ("insert into topics(id, name) values({}, '{}')").format(
                c.id, c.test_name
            )
            cur.execute(sql)
        self.conn.commit()

    def init_rsvp(self):
        filepath = FILE_CONFIG["rsvp"]
        data = pd.read_csv(
            filepath, skipinitialspace=True, escapechar="\\", quotechar='"'
        )
        data["eventid"].astype("str")
        data["mtime"].astype("int")
        cur = self.conn.cursor()
        for row in track(
            data.iterrows(), description="inserting rsvp...", total=len(data)
        ):
            c = row[1]
            sql = (
                "insert into rsvps(eventid, memberid, mtime) values('{}', {}, '{}')"
            ).format(
                c.eventid,
                c.memberid,
                datetime.strftime(
                    datetime.fromtimestamp(c.mtime / 1000), "%Y-%m-%d_%H:%M:%S"
                ),
            )
            cur.execute(sql)
        self.conn.commit()

    def init_groups(self):
        filepath = FILE_CONFIG["groups"]
        data = pd.read_csv(
            filepath, skipinitialspace=True, escapechar="\\", quotechar='"'
        )
        data["id"].astype("int")
        data["rating"].astype("float")
        data["lat"].astype("float")
        data["lon"].astype("float")
        data["created"].astype("int")
        data["members"].astype("int")
        data["organizer"].astype("int")
        cur = self.conn.cursor()
        for row in track(
            data.iterrows(), description="inserting groups...", total=len(data)
        ):
            c = row[1]
            sql = (
                "insert into groups(id, name, urlname, rating, join_mode, lat, lon, country, city, created, members, organizer)"
                "values({}, '{}', '{}', {}, '{}', {}, {}, '{}', '{}', '{}', {}, {})"
            ).format(
                c.id,
                c.test_name,
                c.urlname,
                c.rating,
                c.join_mode,
                c.lat,
                c.lon,
                str(c.country).lower(),
                c.city,
                datetime.strftime(
                    datetime.fromtimestamp(c.created / 1000), "%Y-%m-%d_%H:%M:%S"
                ),
                c.members,
                c.organizer,
            )
            cur.execute(sql)
        self.conn.commit()

    def init_member_groups(self):
        filepath = FILE_CONFIG["member_groups"]
        cursor = self.conn.cursor()
        with open(filepath, "r+") as f:
            data = json.load(f)
            for memberid, values in track(
                data.items(), description="insert into membergroups:"
            ):
                for index, groupid in enumerate(values[0]):
                    sql = (
                        "insert into membergroups(memberid, groupid, join_time) values ({}, {}, '{}')"
                    ).format(
                        int(memberid),
                        int(values[0][index]),
                        datetime.strftime(
                            datetime.fromtimestamp(int(values[1][index]) / 1000),
                            "%Y-%m-%d_%H:%M:%S",
                        ),
                    )
                    cursor.execute(sql)
                    # print(f"sql: {sql}")
        self.conn.commit()

    def get_topic_infos(self, topics):
        """get topic mapping relations

        Args:
            topics (list): a list of topic items, in which each item is [id, name]

        Returns:
            topic_set(dict): a dict, mapping representative topic name to represented topic id list
            topic_mapping(dict): a dictionary mapping topic id to its representative topic id
        """
        topics = self.query("topics", ["id", "name"], ["order by id"])  # [id, 'name']
        # [set, [id]], set represents for word set of name with lower case letter
        topic_info = [
            [
                set([_.lower() for _ in topic_item[1].split()]),
                [topic_item[0]],
                [topic_item[1]],
            ]
            for topic_item in topics
        ]
        topic_info = sorted(topic_info, key=lambda x: len(x[0]))
        # key: topic id, value: id of represents topic
        topic_mapping = dict(
            zip([_[1][0] for _ in topic_info], [_[1][0] for _ in topic_info])
        )

        with Progress() as progress:
            task = progress.add_task(
                "compare topics to find similarity...", total=len(topic_info)
            )
            index = 0
            delete = False
            while True:
                if index >= len(topic_info):
                    break
                item = topic_info[index]
                current_word_set = item[0]  # word set of current item
                for prev_index in range(index):
                    previous_word_set = topic_info[prev_index][0]
                    if (
                        current_word_set & previous_word_set
                    ):  # check whether has common word
                        # append current item to the previous one
                        topic_info[prev_index][1].append(item[1][0])
                        topic_info[prev_index][2].append(item[2][0])
                        topic_mapping[item[1][0]] = topic_info[prev_index][1][0]
                        delete = True
                        break
                if delete:
                    del topic_info[index]
                    delete = False
                else:
                    index += 1
                    progress.update(task, advance=1)
        self.topic_info = topic_info
        self.topic_mapping = topic_mapping
        self.topic_index = dict(
            zip(
                sorted(list(set(topic_mapping.values()))),
                list(range(len(set(topic_mapping.values())))),
            )
        )
        return topic_info, topic_mapping

    def get_edges_by_memberids(self, memberids, members_index):
        """get edges of members to members, members are selected from memberids

        Args:
            memberids (list): a lit of member id
            members_index (dict): dict, key: member id, value: member index in dataset

        Returns:
            edges(tow dimension array): edges[memberi][memberj] = 1 when memberi and memberj are friends or 0 other case
        """
        cursor = self.conn.cursor()
        ids = ", ".join([str(_) for _ in memberids])
        sql = f"select array_agg(rsvps.memberid) from rsvps where memberid in ({ids}) group by eventid"
        if not memberids:
            sql = f"select array_agg(rsvps.memberid) from rsvps group by eventid"
        cursor.execute(sql)
        members_in_same_event = cursor.fetchall()
        members_in_same_event = [_[0] for _ in members_in_same_event]
        edge_index = [[], []]
        for event_members in track(
            members_in_same_event,
            total=len(members_in_same_event),
            description="start to get edges...",
        ):
            for index_i, member_i in enumerate(event_members):
                for index_j in range(index_i + 1, len(event_members)):
                    member_j = event_members[index_j]
                    if (
                        index_i != index_j
                        and int(member_i) in members_index
                        and int(member_j) in members_index
                    ):
                        edge_index[0].append(members_index[int(member_i)])
                        edge_index[1].append(members_index[int(member_j)])
                        edge_index[0].append(members_index[int(member_j)])
                        edge_index[1].append(members_index[int(member_i)])
        self.edge_index = edge_index
        return edge_index

    def get_labels(self):
        cursor = self.conn.cursor()
        labels = [[0 for __ in self.memberids] for _ in self.groupids]
        sql = "select memberid, array_agg(membergroups.groupid) from membergroups group by memberid"
        cursor.execute(sql)
        self.member_groups = cursor.fetchall()
        for memberid, groupids in track(
            self.member_groups, description="start to get label..."
        ):
            for groupid in groupids:
                if memberid in self.member_index and groupid in self.group_index:
                    labels[self.group_index[groupid]][self.member_index[memberid]] = 1
        import numpy as np

        print("each member participated group nums:\n", np.array(labels).sum(axis=0))
        self.labels = labels

    def get_group_features(self):
        if self.simple_topics:  # if use simple topics
            group_features = [
                [0 for __ in range(len(self.topic_info))]
                for _ in range(len(self.groupids))
            ]
            for group, topic in track(
                self.group_topics, description="start to get train group features..."
            ):
                if (
                    group in self.group_index
                    and topic in self.topic_index
                    and topic in self.topic_mapping
                    and self.topic_mapping[topic] in self.topic_index
                ):
                    group_features[self.group_index[group]][
                        self.topic_index[self.topic_mapping[topic]]
                    ] += 1
            self.group_features = group_features
            return
        group_features = [
            [0 for __ in range(len(self.topicids))] for _ in range(len(self.groupids))
        ]
        for group, topic in track(
            self.group_topics, description="start to get train group features..."
        ):
            if group in self.group_index and topic in self.topic_index:
                group_features[self.groups_index[group]][self.topics_index[topic]] = 1
        self.group_features = group_features

    def get_data_by_city(self, min_num_groups=3, citynum=5):
        cursor = self.conn.cursor()
        print("Start get data by city...")
        sql = (
            "select groups.city, "
            "array_agg(distinct groups.id) groupids, "
            "array_agg(distinct members.id) memberids, "
            "array_agg(distinct topicsid) topicids, "
            "array_agg(distinct topics.name) topicnames, "
            "count(distinct groups.id) groupnum, "
            "count(distinct members.id) membernum "
            "from groups left join members on groups.city=members.city "
            "left join membertopics on members.id=membertopics.memberid "
            "left join topics on membertopics.topicsid=topics.id "
            "where topicsid is not null and "
            "topics.name is not null "
            "and members.id in "
            "  (select memberid from "
            "    (select membergroups.memberid, count(distinct groupid) groupnum "
            "     from membergroups  group by memberid "
            "     order by groupnum desc "
            "     ) "
            "     as membergroupnum where groupnum >= {} "
            "  ) "
            "group by groups.city order by groupnum desc limit {}"
        ).format(min_num_groups, citynum)
        cursor.execute(sql)
        data_by_city = cursor.fetchall()
        print("Get data by city finished. Start processing...")
        data_by_city = sorted(data_by_city, key=lambda x: len(x[2]), reverse=True)

        def add_col(x, index):
            result = []
            for item in x:
                result += item[index]
            return result

        self.groupids = list(set(add_col(data_by_city, 1)))
        self.group_topics = sorted(self.query("grouptopics"))
        self.group_index = dict(
            zip(sorted(self.groupids), list(range(len(self.groupids))))
        )
        self.memberids = list(set(add_col(data_by_city, 2)))
        self.member_index = dict(
            zip(sorted(self.memberids), list(range(len(self.memberids))))
        )
        self.topicids = list(set(add_col(data_by_city, 3)))
        # self.topic_index = dict(
        #     zip(sorted(self.topicids), list(range(len(self.topicids)))))
        self.topicnames = add_col(data_by_city, 4)
        # genereate topics info
        ids = ", ".join([str(_) for _ in self.topicids])
        sql = f"select * from topics where id in ({ids}) "
        cursor.execute(sql)
        topics = cursor.fetchall()
        topics = [[_[i] for i in range(len(_))] for _ in topics]
        self.get_edges_by_memberids(self.memberids, self.member_index)
        self.get_topic_infos(topics)
        self.get_labels()
        self.get_group_features()

    def get_data_by_num(self, membernum):
        self.groupids = [_[0] for _ in sorted(self.query("groups", ["id"]))]
        self.topicids = [_[0] for _ in sorted(self.query("topics", ["id"]))]
        self.memberids = [_[0] for _ in sorted(self.query("members", ["id"]))][
            0:membernum
        ]
        self.group_topics = sorted(self.query("grouptopics"))
        self.group_index = dict(zip(self.groupids, range(len(self.groupids))))
        # self.topic_index = dict(
        #     zip(self.topicids, range(len(self.topicids))))
        self.member_index = dict(zip(self.memberids, range(len(self.memberids))))
        self.get_edges_by_memberids([], self.member_index)
        self.topics = self.query("topics", ["id", "name"], ["order by id"])
        self.get_topic_infos(self.topics)
        self.get_labels()
        self.get_group_features()

    def exampleDataFrom(
        self,
        membernum=-1,
        percent=0.8,
        simple_topics=False,
        use_top_city=True,
        citynum=5,
        min_group_num=3,
    ):
        """get train data and prediction data

        Args:
            membernum (int, optional): member num, invaliad when use_top_city=True. Defaults to -1.
            percent (float, optional): train data percent. Defaults to 0.2.
            simple_topics (bool, optional): whether to use reduced topics. Defaults to False.
            use_top_city (bool, optional): whether get data from top member num cities. Defaults to True.
            citynum (int, optional): city number when use_top_city is true. Defaults to 5.
            mini_group_num (int, optional): minimum group number a member supposed to participated. Defaults to 3.

        Returns:
            tuple : [train_dataset, edges], [prediction_dataset, edges]
        """
        self.simple_topics = simple_topics
        self.membernum = membernum
        if use_top_city:
            self.get_data_by_city(min_group_num, citynum)
        else:
            self.get_data_by_num(membernum)

        # generate train data and prediction data
        train_groups = int(len(self.labels) * percent)
        print(f"train data percent: {percent}")
        edges = torch.tensor(self.edge_index, dtype=torch.long)
        train_label = torch.tensor(self.labels[0:train_groups:], dtype=torch.float)
        train_group_features = torch.tensor(
            self.group_features[0:train_groups:], dtype=torch.float
        )
        prediction_group_features = torch.tensor(
            self.group_features[train_groups::], dtype=torch.float
        )
        prediction_label = torch.tensor(self.labels[train_groups::], dtype=torch.float)
        train_dataset = TensorDataset(train_group_features, train_label)
        prediction_dataset = TensorDataset(prediction_group_features, prediction_label)
        print(
            f"train data size: {train_group_features.size()}, prediction data size: {prediction_group_features.size()}"
        )
        # return [train_dataset, edges], [prediction_dataset, edges]
        print(train_group_features.size())
        print(train_label.size())
        print(edges.size())
        groupNum = train_group_features.size()[0]
        predictGroupNum = prediction_group_features.size()[0]
        topicNum = train_group_features.size()[1]
        nodeNum = train_label.size()[1]
        edgeNum = edges.size()[1]

        return [
            train_dataset,
            prediction_dataset,
            edges,
            nodeNum,
            edgeNum,
            topicNum,
            groupNum,
            predictGroupNum,
        ]

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
                    f"Every attributes must be a string, {attr} is {type(attr)}"
                )
        attr_str = ", ".join(attrs) if attrs else " * "
        condition_str = (
            "where "
            if conditions
            and "order by" not in conditions[0]
            and "limit" not in conditions[0]
            else ""
        )
        for index, condition in enumerate(conditions):
            if (
                ("order by" not in condition and "limit" not in condition)
                and (index < len(conditions) - 1)
                and (
                    "order by" not in conditions[index + 1]
                    and "limit" not in conditions[index + 1]
                )
            ):
                condition_str += f" {condition} and "
            elif (
                ("order by" not in condition and "limit" not in condition)
                and (index < len(conditions) - 1)
                and (
                    "order by" in conditions[index + 1]
                    or "limit" in conditions[index + 1]
                )
            ):
                condition_str += f" {condition} "
            elif ("order by" in condition or "limit" in condition) or (
                index == len(conditions) - 1
            ):
                condition_str += f" {condition} "
            else:
                raise ValueError(
                    f"case should not achieved, input error, index: {index}, condition: {condition}"
                )
        sql = f"select {attr_str} from {tablename} {condition_str}"
        cursor = self.conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        result = [[_[i] for i in range(len(_))] for _ in result]
        print(f"sql: {sql}")
        return result

    def __del__(self):
        self.conn.close()


if __name__ == "__main__":
    db = DataBase()
    db.exampleDataFrom(3000, simple_topics=True, use_top_city=True)
    # db.get_topic_sets()
