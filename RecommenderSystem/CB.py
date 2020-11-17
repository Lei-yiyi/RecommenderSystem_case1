#!/usr/bin/env python
# coding: utf-8

# In[55]:


# 代码说明：基于内容的推荐算法的具体实现
# 根据“所有用户对其看过的节目的评分矩阵.xlsx”以及“所有用户看过的节目及所属类型的01矩阵.xlsx”得到用户画像（一个关于评分数据的行向量，
# 表示某个用户对于各个类型的评分），然后将每个用户画像与所有备选推荐节目的画像（一个关于是否含有/具备该类型的01行向量，表示某个节目含
# 有/具备哪些类型）进行相似度的计算，最后将推荐节目集按相似度进行降序排序后取出topN个节目


# In[56]:


import math
import numpy as np
import pandas as pd


# In[57]:


# 创建节目画像
def createItemsProfiles(data_array, labels_names, items_names):
    '''
    data_array: 所有用户看过的节目的所属类型的01矩阵，[[0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [...] ...]
    labels_names：所有类型名，['教育', '戏曲', ...]
    items_names：所有节目名，['大军师司马懿之军师联盟', '非诚勿扰', '记忆大师', ...]
    
    items_profiles：节目画像，{item1:{'label1':1, 'label2': 0, 'label3': 0, ...}, item2:{...}...}
    '''

    items_profiles = {}

    for i in range(len(items_names)):

        items_profiles[items_names[i]] = {}

        for j in range(len(labels_names)):
            items_profiles[items_names[i]][labels_names[j]] = data_array[i][j]

    return items_profiles


# In[58]:


# 创建用户画像
def createUsersProfiles(data_array, users_names, items_names, labels_names, items_profiles):
    '''
    data_array: 所有用户对于其所看过的节目的评分矩阵,[[0.1804 0.042 0.11  0.07  0.19  0.56  0.14  0.3  0.32 0, ...], [...] ...]
    users_names：所有用户名，['A', 'B', 'C']
    items_names：所有节目名，['大军师司马懿之军师联盟', '非诚勿扰', '记忆大师', ...]
    labels_names：所有类型名，['教育', '戏曲', ...]
    items_profiles：节目画像，{item1:{'label1':1, 'label2': 0, 'label3': 0, ...}, item2:{...}...}
    
    users_profiles：用户画像，{user1:{'label1':1.1, 'label2': 0.5, 'label3': 0.0, ...}, user2:{...}...}
    items_users_saw：统计每个用户所看过的节目（不加入隐性评分信息）
    '''

    users_profiles = {}

    # 计算每个用户对所看过的所有节目的平均隐性评分，[0.21248888888888892, 0.19333333333333336, 0.17777777777777778]
    # 统计每个用户所看过的节目（不加入隐性评分信息），{user1:[item1, item3, item5], user2:[...],...}
    # 统计每个用户所看过的节目及评分，{user1:[[item1, 1.1], [item2, 4.1]], user2:...}
    users_average_scores_list = []
    items_users_saw = {}
    items_users_saw_scores = {}

    for i in range(len(users_names)):

        items_users_saw_scores[users_names[i]] = []
        items_users_saw[users_names[i]] = []
        count = 0
        sum = 0.0

        for j in range(len(items_names)):

            # 用户对该节目隐性评分为正，表示真正看过该节目
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

            for item in items_users_saw_scores[users_names[i]]:
                '''
                公式：
                user1_score_to_label1 = Sigma(score_to_itemi - user1_average_score)/items_count
                
                参数：
                user1_score_to_label1：用户user1对于类型label1的隐性评分
                score_to_itemi：用户user1对于其看过的含有类型label1的节目itemi的评分
                user1_average_score：用户user1对其所看过的所有节目的平均评分
                items_count：用户user1看过的节目总数
                '''

                # 该节目含有特定标签labels_names[j]
                if items_profiles[item[0]][labels_names[j]] > 0:
                    score += (item[1] - users_average_scores_list[i])
                    count += 1

            # 如果求出的值太小，直接置0
            if abs(score) < 1e-6:
                score = 0.0
            if count == 0:
                result = 0.0
            else:
                result = score / count

            users_profiles[users_names[i]][labels_names[j]] = result

    return (users_profiles, items_users_saw)


# In[59]:


# 计算用户画像向量与节目画像向量的距离（相似度），向量相似度计算公式：cos(user, item) = sigma_ui/sqrt(sigma_u * sigma_i)
def calCosDistance(user, item, labels_names):
    '''
    user: 某一用户的画像，{'label1':1.1, 'label2': 0.5, 'label3': 0.0, ...}
    item: 某一节目的画像，{'label1':1, 'label2': 0, 'label3': 0, ...}
    labels_names: 所有类型名
    
    sigma_ui/math.sqrt(sigma_u * sigma_i)：用户画像向量与节目画像向量的距离（相似度）
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


# In[60]:


# 基于内容的推荐算法：借助特定某个用户的画像user_profile和备选推荐节目集的画像items_profiles，通过计算向量之间的相似度得出推荐节目集
def contentBased(user_profile, items_profiles, items_names, labels_names, items_user_saw):
    '''
    user_profile: 某一用户的画像，{'label1':1.1, 'label2': 0.5, 'label3': 0.0, ...}
    items_profiles: 备选推荐节目集的节目画像，{item1:{'label1':1, 'label2': 0, 'label3': 0}, item2:{...}...}
    items_names: 备选推荐节目集中的所有节目名
    labels_names: 所有类型名
    items_user_saw: 某一用户看过的节目
    
    recommend_items：按相似度降序排列的推荐节目集，[[节目名, 该节目画像与该用户画像的相似度], ...]
    '''
    
    recommend_items = []

    for i in range(len(items_names)):
        # 从备选推荐节目集中的选择用户user没有看过的节目
        if items_names[i] not in items_user_saw:
            recommend_items.append([items_names[i], calCosDistance(user_profile, items_profiles[items_names[i]], labels_names)])

    # 将推荐节目集按相似度降序排列
    recommend_items.sort(key=lambda item: item[1], reverse=True)

    return recommend_items


# In[61]:


# 输出推荐给该用户的节目列表
def printRecommendedItems(recommend_items_sorted, max_num):
    '''
    recommend_items_sorted：按相似度降序排列的推荐节目集，[[节目名, 该节目画像与该用户画像的相似度], ...]
    max_num：最多输出的推荐节目数，3
    '''
    
    count = 0
    for item, degree in recommend_items_sorted:
        print("节目名：%s， 推荐指数：%f" % (item, degree))
        count += 1
        if count == max_num:
            break


# In[62]:


# 主程序
if __name__ == '__main__':
    
    # 所有用户名
    all_users_names = ['A', 'B', 'C']
    
    # 按指定顺序排列的所有标签（类型）
    all_labels = ['教育', '戏曲', '悬疑', '科幻', '惊悚', '动作', '资讯', '武侠', '剧情', '警匪', '生活', '军事', '言情', '体育', '冒险', '纪实',
                  '少儿教育', '少儿', '综艺', '古装', '搞笑', '广告']
    labels_num = len(all_labels)
    
    # 所有用户对其看过的节目的评分矩阵
    df1 = pd.read_excel("../data/所有用户对其看过的节目的评分矩阵.xlsx")
    (m1, n1) = df1.shape
    data_array1 = np.array(df1.iloc[:m1 + 1, 1:])  # [[0.1804 0.042 0.11  0.07  0.19  0.56  0.14  0.3  0.32 0, ...], [...] ...]
    items_users_saw_names1 = df1.columns[1:].tolist()  # ['大军师司马懿之军师联盟', '非诚勿扰', '记忆大师', ...]
    
    # 所有用户看过的节目及所属类型的01矩阵
    df2 = pd.read_excel("../data/所有用户看过的节目及所属类型的01矩阵.xlsx")
    (m2, n2) = df2.shape
    data_array2 = np.array(df2.iloc[:m2 + 1, 1:])  # [[0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [...] ...]
    items_users_saw_names2 = np.array(df2.iloc[:m2 + 1, 0]).tolist()  # ['大军师司马懿之军师联盟', '非诚勿扰', '记忆大师', ...]

    # 为用户看过的节目建立节目画像items_users_saw_profiles
    items_users_saw_profiles = createItemsProfiles(data_array2, all_labels, items_users_saw_names2)

    # 建立用户画像users_profiles和用户看过的节目集items_users_saw
    (users_profiles, items_users_saw) = createUsersProfiles(data_array1, all_users_names, items_users_saw_names1, all_labels, items_users_saw_profiles)

    # 备选推荐节目集及所属类型01矩阵
    df3 = pd.read_excel("../data/备选推荐节目集及所属类型01矩阵.xlsx")
    (m3, n3) = df3.shape
    data_array3 = np.array(df3.iloc[:m3 + 1, 1:])  # [[0, 0, 0, 0, 0, 1, ...], [...] ...]
    items_to_be_recommended_names = np.array(df3.iloc[:m3 + 1, 0]).tolist()  # ['绑架者', '10月29日探索·发现', '银河护卫队', ...]
    
    # 为备选推荐节目集建立节目画像
    items_to_be_recommended_profiles = createItemsProfiles(data_array3, all_labels, items_to_be_recommended_names)

    for user in all_users_names:
        print("对于用户 %s 的推荐节目如下：" % user)
        # 基于内容的推荐算法：借助特定某个用户的画像user_profile和备选推荐节目集的画像items_profiles，通过计算向量之间的相似度得出推荐节目集
        recommend_items = contentBased(users_profiles[user], items_to_be_recommended_profiles, items_to_be_recommended_names, all_labels, items_users_saw[user])
        # 输出推荐给该用户的节目列表
        printRecommendedItems(recommend_items, 3)
        print() 
