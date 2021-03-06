# RecommenderSystem_case1

原文链接：https://github.com/Yuziquan/RecommenderSystem

## 1、项目目标
实现一个网络电视节目推荐系统，基于每位用户的观看记录以及节目信息，对每位用户实行节目的个性化推荐。

## 2、主要技术
基于用户的协同过滤（UserCF）与基于内容（CB）的推荐算法的后融合。

## 3、实现步骤

    1）基于内容（CB）的推荐算法
    2）基于用户的协同过滤（UserCF）的推荐算法
    3）基于用户的协同过滤（UserCF）与基于内容（CB）的推荐算法的后融合的混合推荐

### 3.1、基于内容（CB）的推荐算法

**a. 刻画节目画像（Item Profiles）**

根据"data/备选推荐节目集及所属类型.xlsx"生成"data/备选推荐节目集及所属类型01矩阵.xlsx"
- [items_labels_to_01matrix.ipynb](RecommenderSystem/items_labels_to_01matrix.ipynb)

**b. 刻画用户画像（User Profiles）并得出推荐集**

根据“用户A\B\C对于其三个月来所看过节目的评分.xls”三个表生成“所有用户对其看过的节目的评分矩阵.xlsx”

根据“用户A\B\C对于其三个月来所看过节目的评分.xls”三个表生成“所有用户看过的节目及所属类型的01矩阵.xlsx”
- [items_saw_labels_to_01matrix.ipynb](RecommenderSystem/items_saw_labels_to_01matrix.ipynb)

根据“所有用户对其看过的节目的评分矩阵.xlsx”以及“所有用户看过的节目及所属类型的01矩阵.xlsx”得到用户画像（一个关于评分数据的行向量，表示某个用户对于各个类型的评分），然后将每个用户画像与所有备选推荐节目的画像（一个关于是否含有/具备该类型的01行向量，表示某个节目含有/具备哪些类型）进行相似度的计算，最后将推荐节目集按相似度进行降序排序后取出topN个节目
- [CB.ipynb](RecommenderSystem/CB.ipynb)

### 3.2、基于用户的协同过滤（UserCF）的推荐算法

**a. 准备好输入数据**

从上面的CB已经得到“备选推荐节目集及所属类型01矩阵.xlsx"和”所有用户对其看过的节目的评分矩阵.xlsx“

**b. 挖掘用户的邻居，得出推荐集**

找出每个用户的k个邻居，结合所有k个邻居看过的节目对该用户进行节目的推荐。注意被推荐的节目应该不包含该用户看过的节目，也不包含不在备选推荐节目集中的节目。（因为备选推荐节目集是商家盈利的节目集，也就是点播收费之类的。而用户观看的节目集包括免费的卫视直播节目以及付费点播的节目，也就是与备选推荐节目集有交集）
- [UserCF.ipynb](RecommenderSystem/UserCF.ipynb)

### 3.3、基于用户的协同过滤（UserCF）与基于内容（CB）的推荐算法的后融合的混合推荐

将CB和userCF的两个推荐集按一定比例混合
- [CB_Mixture_userCF.ipynb](RecommenderSystem/CB_Mixture_userCF.ipynb)
