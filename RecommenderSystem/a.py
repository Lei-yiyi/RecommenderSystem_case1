import sqlalchemy as sqla
import pandas as pd
import numpy as np

# 见文档《小程序数据调研》

engine = sqla.create_engine("mysql+pymysql://root:mysql@10.1.40.199:3376/volunteer")

# wechat_user_log

# 读数据
sql_wul = 'select * from wechat_user_log'
df_wul = pd.read_sql(sql_wul, engine)

df_wul[:5]

df_wul['type'].drop_duplicates()

df_wul['op_type'].drop_duplicates()

df_wul[df_wul['op_type']=='ACTIVITY_REPORT']['type'].drop_duplicates()

df_wul[df_wul['op_type']=='ACTIVITY_REPORT']







# wechat_user

# 读数据
sql_wu = 'select * from wechat_user'
df_wu = pd.read_sql(sql_wu, engine)

df_wu[:5]

# 去'未知用户'
df1 = df_wu[df_wu['username'] != '未知用户'].drop_duplicates()

df1.describe(include = 'all')

df1.columns

# 选取字段
df2 = df1['uid', 'gender', 'org_id', 'org_name',
       'birth_year',
       'address', 
       'register_time', 'deleted',
       'status', 'work_time', 'realed', 'update_time', 'information',
       'service_type', 'service_time', 'expire_start', 'expire_end',
       'province', 'municipal', 'area', 'detail_area', 'education', 'nation',
       'employment_situation', 'personal_strengths', 'province_name',
       'municipal_name', 'area_name', 'has_family', 'bind_project']

df1['gender'].drop_duplicates()













# activity_report

sql_report = 'select id,activity_id,report_uid,evaluate,report_time,browse_num,likes_num,deleted from activity_report'
df_report = pd.read_sql(sql_report, engine)

df_report

sql_activity = 'select id,name,address,organizer_name,description,requirements,status,deleted from activity'
df_activity = pd.read_sql(sql_activity, engine)

df_activity

sql_log = 'select id,name,address,organizer_name,description,requirements,status,deleted from wechat_user_log'
df_log = pd.read_sql(sql_log, engine)

# 1、分析动态信息

df_report.describe(include='all')

df_report['renew'] = int(1)
df_report

df1 = df_report.groupby(['report_uid', 'activity_id']).sum().reset_index().sort_values('deleted')
df1

df1[df1['deleted']==1]

df_report[df_report['activity_id'] == 'f2445784-09b3-433e-a970-ccfc5faa956e'].sort_values('report_time')

# 2、处理动态信息

# 对每一用户的某一动态，求浏览数、点赞数
df1 = df_report.groupby(['report_uid', 'activity_id']).sum().reset_index().drop(columns=['deleted','renew'], axis=1)
df1

# 对每一用户的某一动态，取最新
df_report.sort_values(['report_uid', 'activity_id', 'report_time'],ascending=[0,0,0],inplace=True)
df2 = df_report.groupby(['report_uid', 'activity_id']).head(1)
df2

# 合并以上表,得到动态信息
df3 = pd.merge(df1,df2[['report_uid', 'activity_id', 'report_time', 'evaluate']],
               how='inner',on=['report_uid', 'activity_id'])
df3

# 3、活动标签

# address
df_activity[['address']].drop_duplicates()

# organizer_name
# df_activity[['organizer_name']].drop_duplicates()
dfa = []
for s in df_activity[['organizer_name']].drop_duplicates().values:
    dfa.append(s[0].strip('^[1-9]\d*$'))#('2/01/827/4/6/3/5'))
dfb = pd.DataFrame(dfa).drop_duplicates()
dfb

# name
la = []
for v in df_activity[['name']].values:
    la.extend(v)
la

df_activity[df_activity['name'].str.contains('测试')]



# activity_user

engine = sqla.create_engine('mysql+pymysql://root:mysql@10.1.40.199:3376/volunteer')
sql = 'select uid,activity_id,is_sign_up from activity_user'
df = pd.read_sql(sql, engine)

df.describe(include='all')

# itemcf

import sys
import sqlalchemy as sqla
import pandas as pd
from collections import defaultdict
import math
from operator import itemgetter


class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''
    
    def __init__(self):
        self.trainset = {}
        
        self.n_sim_activity = 20
        self.n_rec_activity = 10
        
        self.activity_sim_mat = {}
        self.activity_popular = {}
        self.movie_count = 0
        
        print('Similar activity number = %d' % self.n_sim_activity, file=sys.stderr)
        print('Recommended movie number = %d' % self.n_rec_activity, file=sys.stderr)
        
    def generate_dataset(self, DBinfo, filename):
        engine = sqla.create_engine(DBinfo)
        sql = 'select uid,activity_id,is_sign_up from ' + filename
        df = pd.read_sql(sql, engine)
        df = df[df['is_sign_up'] == 1].reset_index()

        for i in range(len(df)):
            user, activity, is_sign_up = df['uid'][i], df['activity_id'][i], df['is_sign_up'][i]
            
            self.trainset.setdefault(user, {})
            self.trainset[user][activity] = is_sign_up

    def calc_activity_sim(self):
        ''' calculate activity similarity matrix '''
        for user,activities in self.trainset.items():
            for activity in activities:
                if activity not in self.activity_popular:
                    self.activity_popular[activity] = 0
                self.activity_popular[activity] += 1
                
        self.activity_count = len(self.activity_popular)
        print('total activity number = %d' % self.activity_count, file=sys.stderr)
        
        # count co-rated users between items
        itemsim_mat = self.activity_sim_mat
        
        for user, activities in self.trainset.items():
            for m1 in activities:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in activities:
                    if m1 == m2:
                        continue
                    itemsim_mat[m1][m2] += 1
    
        # calculate similarity matrix
        simfactor_count = 0

        for m1, related_activities in itemsim_mat.items():
            for m2, count in related_activities.items():
                # 余弦相似度
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.activity_popular[m1] * self.activity_popular[m2])
                simfactor_count += 1

        print('Total similarity factor number = %d' %
              simfactor_count, file=sys.stderr)
        
    def recommend(self, user):
        ''' Find K similar activities and recommend N activities '''
        K = self.n_sim_activity
        N = self.n_rec_activity
        rank = {}
        join_activities = self.trainset[user]

        for activity, is_sign_up in join_activities.items():
            for related_activity, similarity_factor in sorted(self.activity_sim_mat[activity].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                rank.setdefault(related_activity, 0)
                rank[related_activity] += similarity_factor * is_sign_up

        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


if __name__ == '__main__':
    
    itemcf = ItemBasedCF()
    
    itemcf.generate_dataset('mysql+pymysql://root:mysql@10.1.40.199:3376/volunteer', 'activity_user')
    
    itemcf.calc_activity_sim()

# 查看某用户推荐结果
user = "07f8084e-ac30-4fce-b774-7aa37adba456"
print("推荐结果", itemcf.recommend(user))

# CB activities

# 这里的物品即活动
# 这里的评分即对活动的浏览、点赞、评论、收藏、加入
# 这里的标签即活动类型


import sqlalchemy as sqla
import pandas as pd
import re
import numpy as np
import math


def createLabelMatrix(labels_names):
    '''
    创建所有用户看过的物品的所属标签的01矩阵、所有物品id
    
    data_array: 所有用户看过的物品的所属标签的01矩阵，[[0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [...] ...]
    items_names：所有物品id，['0001d4db-bbb0-46fe-99dc-97c6c6954ae7', '004117a3-2ff0-408c-bf44-d9fd9026a60c', ...]
    '''

    # 为所有用户看过的物品打标签
    engine = sqla.create_engine("mysql+pymysql://root:mysql@10.1.40.199:3376/volunteer")
    act = pd.read_sql("select id, name, address, organizer_name, description from activity", engine)

    df["name"] = df["name"].map(str) + df["description"].map(str)
    df = df[["id", "name", "address", "organizer_name"]]
    
    for i in labels_names:
        df[i] = 0
        
    def itemsClassification(data):
        if re.search(r"(每(周|月|天|日)|日常)", data["name"]):
            data["日常"] = 1
            data["其他(大)"] = 1
        if re.search(r"跑", data["name"]):
            data["跑步"] = 1
            data["健身"] = 1
        if re.search(r"瑜伽", data["name"]):
            data["瑜伽"] = 1
            data["健身"] = 1
        if re.search(r"(敬老|老|长者)", data["name"]):
            data["敬老"] = 1
            data["公益"] = 1
        if re.search(r"(扶贫|贫)", data["name"]):
            data["扶贫"] = 1
            data["公益"] = 1
        if re.search(r"(助残|残)", data["name"]):
            data["助残"] = 1
            data["公益"] = 1
        if re.search(r"助学", data["name"]):
            data["助学"] = 1
            data["公益"] = 1
        if re.search(r"(环保|家|家园|环境)", data["name"]):
            data["环保"] = 1
            data["公益"] = 1
        if re.search(r"助学", data["name"]):
            data["社区服务"] = 1
            data["公益"] = 1
        if re.search(r"(公益|爱心|宣传|讲座|交通|指引|礼)", data["name"]):
            data["公益宣传"] = 1
            data["公益"] = 1
        if re.search(r"(歌咏|歌|唱|歌唱|唱歌|咏|诵|朗诵)", data["name"]):
            data["歌咏"] = 1
            data["文艺"] = 1
        if re.search(r"(舞蹈|跳舞|舞)", data["name"]):
            data["舞蹈"] = 1
            data["文艺"] = 1
        if re.search(r"画", data["name"]):
            data["绘画"] = 1
            data["文艺"] = 1
        if re.search(r"(书法|练字|字)", data["name"]):
            data["书法"] = 1
            data["文艺"] = 1
        if re.search(r"(戏曲|戏)", data["name"]):
            data["戏曲"] = 1
            data["文艺"] = 1
        if re.search(r"花灯", data["name"]):
            data["花灯"] = 1
            data["文艺"] = 1
        if re.search(r"(旗袍|旗袍秀)", data["name"]):
            data["旗袍秀"] = 1
            data["文艺"] = 1
        if re.search(r"(戏曲|戏)", data["name"]):
            data["戏曲"] = 1
            data["文艺"] = 1
        if re.search(r"(巡查|查|巡)", data["name"]):
            data["社区服务"] = 1
            data["公益"] = 1
        if re.search(r"(阅读|读|书|文学)", data["name"]):
            data["文学"] = 1
            data["文艺"] = 1
        if re.search(r"(快走|散步)", data["name"]):
            data["快走"] = 1
            data["健身"] = 1
        if re.search(r"(骑行|自行车|单车)", data["name"]):
            data["骑行"] = 1
            data["健身"] = 1
        if re.search(r"(爬山|登山|山)", data["name"]):
            data["爬山"] = 1 
            data["健身"] = 1
        if re.search(r"太极", data["name"]):
            data["太极"] = 1
            data["健身"] = 1
        if re.search(r"气功", data["name"]):
            data["气功"] = 1
            data["健身"] = 1
        if re.search(r"广场舞", data["name"]):
            data["广场舞"] = 1
            data["健身"] = 1
        if re.search(r"瑜伽", data["name"]):
            data["瑜伽"] = 1
            data["健身"] = 1
        if re.search(r"(亲子|折纸|阅读|野餐|郊游|手)", data["name"]):
            data["亲子"] = 1
            data["亲子(大)"] = 1
        if re.search(r"(折纸|手|陶艺|陶|剪纸)", data["name"]):
            data["手工"] = 1 
            data["文艺"] = 1
        if re.search(r"(议事|会议|议)", data["name"]):
            data["议事"] = 1
            data["社区发展"] = 1
        if re.search(r"服务", data["name"]):
            data["服务"] = 1 
            data["社区发展"] = 1
        if sum(data[4:]) == 0:
            data["其他"] = 1
            data["其他(大)"] = 1
        return data
    
    df = df.apply(itemsClassification, axis=1)
    
    # 所有用户看过的物品的所属标签的01矩阵、所有物品id
    items_names = df['id'].values
    df.drop(columns=['id', 'name', 'address', 'organizer_name'], inplace=True)
    data_array = df.values
    
    return data_array,items_names

# CB reports

# 这里的物品即报道
# 这里的评分即对报道的浏览、点赞、评论，评分是这三个的数目总和
# 这里的标签即动态类型（动态包括报道和团队相册，报道类型通过活动类型间接获得，团队相册类型通过团队类型间接获得）

import sqlalchemy as sqla
import pandas as pd
import re
import numpy as np
import math


def createLabelMatrix(labels_names):
    '''
    创建所有物品的所属标签的01矩阵、所有物品id
    
    data_array: 所有物品的所属标签的01矩阵，[[0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [...] ...]
    items_names：所有物品id，['0001d4db-bbb0-46fe-99dc-97c6c6954ae7', '004117a3-2ff0-408c-bf44-d9fd9026a60c', ...]
    '''

    # 为物品打标签
    engine = sqla.create_engine("mysql+pymysql://root:mysql@10.1.40.199:3376/volunteer")
    df = pd.read_sql("""SELECT ar.id, a.name, a.description 
                     FROM activity_report ar, activity a 
                     WHERE a.id=ar.activity_id""", engine)

    df["name"] = df["name"].map(str) + df["description"].map(str)
    df = df[["id", "name"]]
    
    for i in labels_names:
        df[i] = 0
        
    def itemsClassification(data):
        if re.search(r"(每(周|月|天|日)|日常)", data["name"]):
            data["日常"] = 1
            data["其他(大)"] = 1
        if re.search(r"跑", data["name"]):
            data["跑步"] = 1
            data["健身"] = 1
        if re.search(r"瑜伽", data["name"]):
            data["瑜伽"] = 1
            data["健身"] = 1
        if re.search(r"(敬老|老|长者)", data["name"]):
            data["敬老"] = 1
            data["公益"] = 1
        if re.search(r"(扶贫|贫)", data["name"]):
            data["扶贫"] = 1
            data["公益"] = 1
        if re.search(r"(助残|残)", data["name"]):
            data["助残"] = 1
            data["公益"] = 1
        if re.search(r"助学", data["name"]):
            data["助学"] = 1
            data["公益"] = 1
        if re.search(r"(环保|家|家园|环境)", data["name"]):
            data["环保"] = 1
            data["公益"] = 1
        if re.search(r"助学", data["name"]):
            data["社区服务"] = 1
            data["公益"] = 1
        if re.search(r"(公益|爱心|宣传|讲座|交通|指引|礼)", data["name"]):
            data["公益宣传"] = 1
            data["公益"] = 1
        if re.search(r"(歌咏|歌|唱|歌唱|唱歌|咏|诵|朗诵)", data["name"]):
            data["歌咏"] = 1
            data["文艺"] = 1
        if re.search(r"(舞蹈|跳舞|舞)", data["name"]):
            data["舞蹈"] = 1
            data["文艺"] = 1
        if re.search(r"画", data["name"]):
            data["绘画"] = 1
            data["文艺"] = 1
        if re.search(r"(书法|练字|字)", data["name"]):
            data["书法"] = 1
            data["文艺"] = 1
        if re.search(r"(戏曲|戏)", data["name"]):
            data["戏曲"] = 1
            data["文艺"] = 1
        if re.search(r"花灯", data["name"]):
            data["花灯"] = 1
            data["文艺"] = 1
        if re.search(r"(旗袍|旗袍秀)", data["name"]):
            data["旗袍秀"] = 1
            data["文艺"] = 1
        if re.search(r"(戏曲|戏)", data["name"]):
            data["戏曲"] = 1
            data["文艺"] = 1
        if re.search(r"(巡查|查|巡)", data["name"]):
            data["社区服务"] = 1
            data["公益"] = 1
        if re.search(r"(阅读|读|书|文学)", data["name"]):
            data["文学"] = 1
            data["文艺"] = 1
        if re.search(r"(快走|散步)", data["name"]):
            data["快走"] = 1
            data["健身"] = 1
        if re.search(r"(骑行|自行车|单车)", data["name"]):
            data["骑行"] = 1
            data["健身"] = 1
        if re.search(r"(爬山|登山|山)", data["name"]):
            data["爬山"] = 1 
            data["健身"] = 1
        if re.search(r"太极", data["name"]):
            data["太极"] = 1
            data["健身"] = 1
        if re.search(r"气功", data["name"]):
            data["气功"] = 1
            data["健身"] = 1
        if re.search(r"广场舞", data["name"]):
            data["广场舞"] = 1
            data["健身"] = 1
        if re.search(r"瑜伽", data["name"]):
            data["瑜伽"] = 1
            data["健身"] = 1
        if re.search(r"(亲子|折纸|阅读|野餐|郊游|手)", data["name"]):
            data["亲子"] = 1
            data["亲子(大)"] = 1
        if re.search(r"(折纸|手|陶艺|陶|剪纸)", data["name"]):
            data["手工"] = 1 
            data["文艺"] = 1
        if re.search(r"(议事|会议|议)", data["name"]):
            data["议事"] = 1
            data["社区发展"] = 1
        if re.search(r"服务", data["name"]):
            data["服务"] = 1 
            data["社区发展"] = 1
        if sum(data[4:]) == 0:
            data["其他"] = 1
            data["其他(大)"] = 1
        return data
    
    df = df.apply(itemsClassification, axis=1)
    
    # 所有物品的所属标签的01矩阵、所有物品id
    items_names = df['id'].values
    df.drop(columns=['id', 'name'], inplace=True)
    data_array = df.values
    
    return data_array,items_names

   
def createItemsProfiles(data_array, labels_names, items_names):
    '''
    创建物品画像
    
    data_array: 所有物品的所属标签的01矩阵，[[0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [...] ...]
    labels_names：所有标签，['敬老', '扶贫', ...]
    items_names：所有物品id，['0001d4db-bbb0-46fe-99dc-97c6c6954ae7', '004117a3-2ff0-408c-bf44-d9fd9026a60c', ...]
    
    items_profiles：物品画像，{item1:{'label1':1, 'label2': 0, 'label3': 0, ...}, item2:{...}...}
    '''
    
    items_profiles = {}
    
    for i in range(len(items_names)):

        items_profiles[items_names[i]] = {}

        for j in range(len(labels_names)):
            items_profiles[items_names[i]][labels_names[j]] = data_array[i][j]

    return items_profiles


def createRatingMatrix():
    '''
    创建所有用户对于其所看过的物品的评分矩阵、所有用户id、所有物品id
    
    data_array：所有用户对于其所看过的物品的评分矩阵,
                [[0.1804 0.042 0.11  0.07  0.19  0.56  0.14  0.3  0.32 0, ...], [...] ...]
    users_names：所有用户id，['00eaa73c-9623-42be-8d81-0357dfcab777', '07f8084e-ac30-4fce-b774-7aa37adba456', ...]
    items_names：所有物品id，['0001d4db-bbb0-46fe-99dc-97c6c6954ae7', '004117a3-2ff0-408c-bf44-d9fd9026a60c', ...]
    '''
    
    def reportsUserScores():
        '''
        统计所有用户对于其所看过的物品的评分 —— 报道
        '''
        engine = sqla.create_engine("mysql+pymysql://root:mysql@10.1.40.199:3376/volunteer")
        
        report_browse_num_sql = """select log.uid, business_id activity_report_id, count(business_id) browse_num
                                from activity_report ar
                                RIGHT JOIN 
                                (SELECT * FROM wechat_user_log WHERE type='ENTER_REPORT_DETAIL') log
                                ON ar.id=log.business_id
                                GROUP BY log.uid, business_id"""
        report_browse_num = pd.read_sql(report_browse_num_sql, engine)
        
        report_like_num = pd.read_sql("SELECT activity_report_id, uid, isLike FROM activity_report_user WHERE isLike = 1", engine)
        
        _ = pd.merge(report_like_num, report_browse_num, on=["uid","activity_report_id"], how="outer").fillna(0)
        
        report_eval_num_sql = """SELECT av.eval_uid uid, av.activity_id activity_report_id, count(av.id) eval_num
                            FROM activity_eval av
                            LEFT JOIN activity_report ar
                            ON av.activity_id=ar.id
                            GROUP BY av.eval_uid, av.activity_id
                            HAVING av.activity_id IS NOT NULL
                            """
        report_eval_num = pd.read_sql(report_eval_num_sql, engine)
        
        reports_sum = pd.merge(report_eval_num, _, on=["uid","activity_report_id"], how="outer").fillna(0)
        
        reports_sum["rank"] = reports_sum["eval_num"] + reports_sum["isLike"] + reports_sum["browse_num"]
        reports_sum.drop(columns=['eval_num', 'isLike', 'browse_num'], inplace=True)
        
        reports_sum.rename(columns={'activity_report_id':'itemid'},inplace=True)

        return reports_sum
    
    def team_album_reports():
        '''
        统计所有用户对于其所看过的物品的评分 —— 团队相册
        '''
        
        engine = sqla.create_engine("mysql+pymysql://root:mysql@10.1.40.199:3376/volunteer")
        
        album_browse_num_sql = """SELECT uid, business_id `id`, count(id) browse_num
                                FROM wechat_user_log 
                                WHERE business_id IN (SELECT id FROM team_album) AND type='ENTER_REPORT_DETAIL'
                                GROUP BY uid, business_id"""
        album_browse_num = pd.read_sql(album_browse_num_sql, engine)
        album_like_num_sql = """SELECT team_album_id `id`, uid, is_like FROM team_album_user WHERE is_like = 1"""
        album_like_num = pd.read_sql(album_like_num_sql, engine)
        album_eval_num_sql = """SELECT te.eval_uid `uid`, te.album_id `id`, count(id) eval_num
                                FROM team_album_eval te
                                GROUP BY te.eval_uid, te.album_id
                                HAVING te.album_id IS NOT NULL"""
        album_eval_num = pd.read_sql(album_eval_num_sql, engine)
        _ = pd.merge(album_browse_num, album_like_num, on=["id", "uid"], how="outer").fillna(0)
        album_sum = pd.merge(album_eval_num, _, on=["uid", "id"], how="outer").fillna(0)
        album_sum["rank"] = album_sum["browse_num"] + album_sum["is_like"] + album_sum["eval_num"]
        album_sum.rename(columns={'id':'itemid'},inplace=True)
        return album_sum[["uid", "itemid", "rank"]]
     
    reports_sum = reportsUserScores()
    album_sum = team_album_reports()
    dynamic_scores = reports_sum.append(reports_sum)

    # 一维变二维
    dimension = reports_sum.pivot('uid','itemid','rank').reset_index().fillna(0)
    
    # 所有用户id
    users_names = dimension['uid'].values
    
    # 所有物品id
    items_names = dimension.keys()[1:]
    
    dimension.drop(columns=['uid'], inplace=True)
    
    # 所有用户对于其所看过的物品的评分矩阵
    data_array = dimension.values
    
    return data_array,users_names,items_names


def createUsersProfiles(data_array, users_names, items_names, labels_names, items_profiles):
    '''
    创建用户画像、用户看过的物品集
    
    data_array: 所有用户对于其所看过的物品的评分矩阵,[[0.1804 0.042 0.11  0.07  0.19  0.56  0.14  0.3  0.32 0, ...], [...] ...]
    users_names：所有用户id，['00eaa73c-9623-42be-8d81-0357dfcab777', '07f8084e-ac30-4fce-b774-7aa37adba456', ...]
    items_names：所有物品id，['0001d4db-bbb0-46fe-99dc-97c6c6954ae7', '004117a3-2ff0-408c-bf44-d9fd9026a60c', ...]
    labels_names：所有标签，['敬老', '扶贫', ...]
    items_profiles：物品画像，{item1:{'label1':1, 'label2': 0, 'label3': 0, ...}, item2:{...}...}
    
    users_profiles：用户画像，{user1:{'label1':1.1, 'label2': 0.5, 'label3': 0.0, ...}, user2:{...}...}
    items_users_saw：用户看过的物品集（不加入隐性评分信息），{user1: ['item1', 'item1' ...], user2:[...]...}
    '''

    users_profiles = {}

    # 计算每个用户对所看过的所有物品的平均隐性评分，[0.21248888888888892, 0.19333333333333336, 0.17777777777777778]
    # 统计每个用户所看过的物品（不加入隐性评分信息），{user1:[item1, item3, item5], user2:[...],...}
    # 统计每个用户所看过的物品及评分，{user1:[[item1, 1.1], [item2, 4.1]], user2:...}
    users_average_scores_list = []
    items_users_saw = {}
    items_users_saw_scores = {}

    for i in range(len(users_names)):

        items_users_saw_scores[users_names[i]] = []
        items_users_saw[users_names[i]] = []
        count = 0
        sum = 0.0

        for j in range(len(items_names)):

            # 用户对该物品隐性评分为正，表示真正看过该物品
            if data_array[i][j] > 0:
                items_users_saw[users_names[i]].append(items_names[j])
                items_users_saw_scores[users_names[i]].append([items_names[j], data_array[i][j]])
                count += 1
                sum += data_array[i][j]

        if count == 0:
            users_average_scores_list.append(0)
        else:
            users_average_scores_list.append(sum / count)

    # 用户画像，{user1:{'label1':1.1, 'label2': 0.5, 'label3': 0.0, ...}, user2:{...}...}
    for i in range(len(users_names)):

        users_profiles[users_names[i]] = {}

        for j in range(len(labels_names)):
            count = 0
            score = 0.0

#             for item in items_users_saw_scores[users_names[i]]:
#                 '''
#                 公式：
#                 user1_score_to_label1 = Sigma(score_to_itemi - user1_average_score)/items_count
                
#                 参数：
#                 user1_score_to_label1：用户user1对于标签label1的隐性评分
#                 score_to_itemi：用户user1对于其看过的含有标签label1的物品itemi的评分
#                 user1_average_score：用户user1对其所看过的所有物品的平均评分
#                 items_count：用户user1看过的物品总数
#                 '''

#                 # 该物品含有特定标签labels_names[j]
#                 if items_profiles[item[0]][labels_names[j]] > 0:
#                     score += (item[1] - users_average_scores_list[i])
#                     count += 1

            # 如果求出的值太小，直接置0
            if abs(score) < 1e-6:
                score = 0.0
            if count == 0:
                result = 0.0
            else:
                result = score / count

            users_profiles[users_names[i]][labels_names[j]] = result

    return users_profiles, items_users_saw


def calCosDistance(user, item, labels_names):
    '''
    计算用户画像向量与物品画像向量的距离（相似度），向量相似度计算公式：cos(user, item) = sigma_ui/sqrt(sigma_u * sigma_i)
    
    user: 某一用户的画像，{'label1':1.1, 'label2': 0.5, 'label3': 0.0, ...}
    item: 某一物品的画像，{'label1':1, 'label2': 0, 'label3': 0, ...}
    labels_names: 所有标签，['敬老', '扶贫', ...]
    
    sigma_ui/math.sqrt(sigma_u * sigma_i)：用户画像向量与物品画像向量的距离（相似度）
    '''

    sigma_ui = 0.0
    sigma_u = 0.0
    sigma_i = 0.0

    for label in labels_names:
        sigma_ui += user[label] * item[label]
        sigma_u += (user[label] * user[label])
        sigma_i += (item[label] * item[label])

    if sigma_u == 0.0 or sigma_i == 0.0:  # 若分母为0，相似度为0
        return 0

    return sigma_ui/math.sqrt(sigma_u * sigma_i)


def contentBased(user_profile, items_profiles, items_names, labels_names, items_user_saw):
    '''
    借助某个用户的画像user_profile和备选推荐物品集的画像items_profiles，通过计算向量之间的相似度得出推荐物品集

    user_profile: 某一用户的画像，{'label1':1.1, 'label2': 0.5, 'label3': 0.0, ...}
    items_profiles: 备选推荐物品集的物品画像，{item1:{'label1':1, 'label2': 0, 'label3': 0}, item2:{...}...}
    items_names: 备选推荐物品集中的所有物品id
    labels_names: 所有标签
    items_user_saw: 某一用户看过的物品
    
    recommend_items：按相似度降序排列的推荐物品集，[[物品名, 该物品画像与该用户画像的相似度], ...]
    '''
    
    recommend_items = []

    for i in range(len(items_names)):
        # 从备选推荐物品集中的选择用户user没有看过的物品
        if items_names[i] not in items_user_saw:
            recommend_items.append([items_names[i], calCosDistance(user_profile, items_profiles[items_names[i]], labels_names)])

    # 将推荐物品集按相似度降序排列
    recommend_items.sort(key=lambda item: item[1], reverse=True)

    return recommend_items


def printRecommendedItems(recommend_items_sorted, max_num):
    '''
    输出推荐给该用户的物品列表
    
    recommend_items_sorted：按相似度降序排列的推荐物品集，[[物品名, 该物品画像与该用户画像的相似度], ...]
    max_num：最多输出的推荐物品数，3
    '''
    
    count = 0
    for item, degree in recommend_items_sorted:
        print("物品id：%s， 推荐指数：%f" % (item, degree))
        count += 1
        if count == max_num:
            break


# 主程序
if __name__ == '__main__':
    
    # 用户id
    user = '00eaa73c-9623-42be-8d81-0357dfcab777'
    
    # 按指定顺序排列的所有标签
    label_dict = {"公益": ['敬老', '扶贫', '助残', '助学', '环保', '社区服务', '公益宣传'],
                  "文艺": ['歌咏', '舞蹈','绘画', '书法', '戏曲', '文学', '花灯', '旗袍秀', "手工"],
                  "健身": ['跑步', '快走', '骑行', '爬山', '太极','气功', '广场舞', '瑜伽'],
                  "亲子(大)": ['亲子'],
                  "社区发展": ['议事', '服务'],
                  "其他(大)": ["其他", "日常"]}
    labels_names = sum(label_dict.values(), []) + list(label_dict.keys())
    
    # 创建所有物品的所属标签的01矩阵、所有物品id
    all_itemsLabelMatrix,all_itemsID = createLabelMatrix(labels_names) 
    
    # 创建所有用户对于其所看过的物品的评分矩阵、所有用户id、所有物品id
    saw_itemsRatingMatrix,users_names,saw_itemsID = createRatingMatrix()
    
    # 创建用户看过的物品的所属标签的01矩阵
#     saw_itemsLabelMatrix = []
#     for item in saw_itemsID:
#         saw_itemsLabelMatrix.append(all_itemsLabelMatrix[all_itemsID.index(item)])


    # ------------------------------------- 为用户看过的物品创建物品画像 ------------------------------------ #    
    

#     # 为用户看过的物品创建物品画像items_users_saw_profiles
#     items_users_saw_profiles = createItemsProfiles(data_users_saw_array_lable, labels_names, items_users_saw_names_lable)
    
    # ------------------------------------------ 创建用户画像 ----------------------------------------------- # 
    
        
#     # 创建用户画像users_profiles、用户看过的物品集items_users_saw
#     users_profiles, items_users_saw = createUsersProfiles(data_users_saw_array_rating, users_names, 
#                                                           items_users_saw_names_rating, labels_names, items_users_saw_profiles)
       
    # ----------------------------------- 为备选推荐物品集创建物品画像 -------------------------------------- #  
    
#     # 创建备选推荐物品的所属标签的01矩阵、所有物品id
#     recommend_itemsLabelMatrix,recommend_itemsID = createLabelMatrix(labels_names)
#     # 为备选推荐物品集创建物品画像
#     items_to_be_recommended_profiles = createItemsProfiles(data_to_be_recommended_array_lable, labels_names, 
#                                                            items_to_be_recommended_names_lable)
    
    # ---------------------------------------------- 推荐算法 ----------------------------------------------- # 

#     print("对于用户 %s 的推荐物品如下：" % user)
#     # 借助特定某个用户的画像user_profile和备选推荐物品集的画像items_profiles，通过计算向量之间的相似度得出推荐物品集
#     recommend_items = contentBased(users_profiles[user], items_to_be_recommended_profiles, items_to_be_recommended_names_lable, labels_names, items_users_saw[user])
#     # 输出推荐给该用户的物品列表
#     printRecommendedItems(recommend_items, 3)

saw_itemsLabelMatrix = []
# all_itemsID_ = list(enumerate(all_itemsID))
# all_itemsID_[0]
# 
# for k, v in enumerate(saw_itemsID):
#     print(k, v)
    
all_itemsID_ = {v:k for k, v in enumerate(all_itemsID)}
for item in saw_itemsID:
    print(all_itemsLabelMatrix[all_itemsID_[item]])

# print(saw_itemsLabelMatrix.append(all_itemsLabelMatrix[0]))

all_itemsID_['007912f1-5739-4468-be5f-261bc9e39fab']

np.where(all_itemsID=='02f7b0f3-328f-4e2b-a771-522920d87bb6')

saw_itemsLabelMatrix

np.where(all_itemsLabelMatrix=='007912f1-5739-4468-be5f-261bc9e39fab')

df = pd.DataFrame(
{
    'key1':['one','two','three','one','two','four'],
    'key2':['A','B','C','D','C','D'],
    'value':np.random.randn(6)
}
)
df

dimension = df.pivot('key1','key2','value').reset_index().fillna(0)
# dimension.drop(columns=['key2'], inplace = True)
# dimension.columns
dimension

# data_array: 所有用户对于其所看过的物品的评分（对报道的浏览、点赞、评论）矩阵,[[0.1804 0.042 0.11  0.07  0.19  0.56  0.14  0.3  0.32 0, ...], [...] ...]
dimension.keys()[1:]

dimension.drop(columns=['key1'], inplace=True)
dimension

dimension.values





# get_reports--hxr

import pandas as pd
import sqlalchemy as sqla
from collections import defaultdict
import math
from operator import itemgetter


class ItemBasedCF(object):

    """ TopN recommendation - Item Based Collaborative Filtering """

    def __init__(self, n_sim_activity=30, n_rec_activity=20):
        self.trainset = {}

        self.n_sim_activity = n_sim_activity
        self.n_rec_activity = n_rec_activity

        self.activity_sim_mat = {}
        self.activity_popular = {}
        self.movie_count = 0

    #         print('Similar activity number = %d' % self.n_sim_activity, file=sys.stderr)
    #         print('Recommended movie number = %d' % self.n_rec_activity, file=sys.stderr)

    # a: user, b: activity
    def generate_dataset(self, df, aid, bid, score):
        for i in range(len(df)):
            a, b, rank = df[aid][i], df[bid][i], df[score][i]

            self.trainset.setdefault(a, {})
            self.trainset[a][b] = rank

    def calc_activity_sim(self):
        """ calculate activity similarity matrix """
        for user, activities in self.trainset.items():
            for activity in activities:
                if activity not in self.activity_popular:
                    self.activity_popular[activity] = 0
                self.activity_popular[activity] += 1

        self.activity_count = len(self.activity_popular)
        #         print('total activity number = %d' % self.activity_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.activity_sim_mat

        for user, activities in self.trainset.items():
            for m1 in activities:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in activities:
                    if m1 == m2:
                        continue
                    itemsim_mat[m1][m2] += 1

        # calculate similarity matrix
        simfactor_count = 0

        for m1, related_activities in itemsim_mat.items():
            for m2, count in related_activities.items():
                # 余弦相似度
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.activity_popular[m1] * self.activity_popular[m2])
                simfactor_count += 1

    #         print('Total similarity factor number = %d' %
    #               simfactor_count, file=sys.stderr)

    def recommend(self, user):
        """ Find K similar activities and recommend N activities """
        K = self.n_sim_activity
        N = self.n_rec_activity
        rank = {}
        try:
            join_activities = self.trainset[user]

            for activity, ranks in join_activities.items():
                for related_activity, similarity_factor in sorted(self.activity_sim_mat[activity].items(),
                                                                  key=itemgetter(1), reverse=True)[:K]:
                    rank.setdefault(related_activity, 0)
                    rank[related_activity] += similarity_factor * ranks

            return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
        except KeyError:
            return None


class ReportsRecommend(object):
    def __init__(self, uid):
        """
        :param uid: 用户id
        """
        # 保存了所有动态
        self.engine = sqla.create_engine("mysql+pymysql://root:mysql@10.1.40.149:3376/new_bak")
        # 限制300条动态
        report_limit = 300
        reports_sql = f"""SELECT * FROM activity_report
        ORDER BY report_time DESC, likes_num DESC, browse_num DESC
        LIMIT {report_limit}
        """
        self.all_reports = pd.read_sql(reports_sql, self.engine)
        # 相似活动数量/推荐活动数量
        similar_num, rec_num = 50, 50
        # 基于动态的协同过滤
        self.item_cf = ItemBasedCF(similar_num, rec_num)
        self.item_cf.generate_dataset(self.reports_user_scores(), "uid", "activity_report_id", "rank")
        self.item_cf.calc_activity_sim()
        # 当前用户的所有推荐
        self.recommends = self.reports_recommend(uid)
        
    def reports_user_scores(self):
        # 统计动态-用户的评分（包括喜欢/浏览/评论）
        # 分数是这三个的总和，直接相加，没权重
        report_browse_num_sql = """select log.uid, business_id activity_report_id, count(business_id) browse_num
                                from activity_report ar
                                RIGHT JOIN 
                                (SELECT * FROM wechat_user_log WHERE type='ENTER_REPORT_DETAIL') log
                                ON ar.id=log.business_id
                                GROUP BY log.uid, business_id"""
        report_browse_num = pd.read_sql(report_browse_num_sql, self.engine)
        report_like_num = pd.read_sql("SELECT activity_report_id, uid, isLike FROM activity_report_user WHERE isLike = 1", self.engine)
        # 连接like和browse
        _ = pd.merge(report_like_num, report_browse_num, on=["uid","activity_report_id"], how="outer").fillna(0)
        report_eval_num_sql = """SELECT av.eval_uid uid, av.activity_id activity_report_id, count(av.id) eval_num
                            FROM activity_eval av
                            LEFT JOIN activity_report ar
                            ON av.activity_id=ar.id
                            GROUP BY av.eval_uid, av.activity_id
                            HAVING av.activity_id IS NOT NULL
                            """
        report_eval_num = pd.read_sql(report_eval_num_sql, self.engine)
        reports_sum = pd.merge(report_eval_num, _, on=["uid","activity_report_id"], how="outer").fillna(0)
        reports_sum["rank"] = reports_sum["eval_num"] + reports_sum["isLike"] + reports_sum["browse_num"]
        return reports_sum

    def reports_recommend(self, uid):
        rec = self.item_cf.recommend(uid)
        # 如果当前用户没有动态记录，直接随机最新的动态
        if rec:
            # 推荐的动态id
            rec = pd.DataFrame([i[0] for i in rec], columns=["id"])
            rec = pd.merge(rec, self.all_reports, on="id", how="inner")
            # 非推荐动态
            left = pd.concat([self.all_reports, rec, rec]).drop_duplicates(keep=False).sample(frac=1)
        else:
            # 说明该用户没有动态记录, 直接推荐排序后的所有动态
            rec = None
            # 打乱顺序
            left = self.all_reports.sample(frac=1)
        return pd.concat([rec, left])
    
    def get_reports(self, page):
        """
        :param page: 动态页数，从1开始
        :return: 当前页数的动态列表
        """
        reports_amount = 300
        per_page = 15
        page_amount = reports_amount / per_page
        return self.recommends[(page-1)*per_page:page*per_page].sample(n=10).to_json(orient="table",force_ascii=False)
    
    
if __name__ == "__main__":
    # 每个用户一个实例，按页数拿动态列表
    rr = ReportsRecommend("3d35f98b-bd9f-4e48-8bf0-f46cf7d83770")  # 用户id
    print(rr.get_reports(1))  # 页数

