[如期而至-用户购买时间预测](https://jdata.jd.com/html/detail.html?id=2)

最终名次为第66名。

这次准备时间不是很多，构造了一些基本的特征，用lgb模型。

这次主要的难点在数据集划分上，我的方法基本上能做到线上线下同增同减，而且A/B也比较稳定。

划分思路为：由于预测的时3个月内购买过的用户，下个月是否购买以及购买的时间。因为划分时需要那3个月来统计特征，第4个月做label。


A榜时简单做了几次，B榜时直接用A榜的模型chenxl_b文件中，就得到66名的成绩。其实还用SBB做了一版（在chenxl_b_stack中），但是忘了提交了。。。


