#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:35:06 2018

@author: chenxl
"""

import numpy as np
import pandas as pd

#from tqdm import tqdm

def read_data(path):
    #表1：SKU基本信息表（jdata_sku_basic_info）
    sku_basic_info = pd.read_csv(path+"jdata_sku_basic_info.csv")
    #表2：用户基本信息表（jdata_user_basic_info）
    user_basic_info = pd.read_csv(path+"jdata_user_basic_info.csv")
    #表3：用户行为表（jdata_user_action）
    user_action = pd.read_csv(path+"jdata_user_action.csv", parse_dates=['a_date']) #时间太长
    #表4：用户订单表（jdata_user_order）
    user_order = pd.read_csv(path+"jdata_user_order.csv",parse_dates=['o_date'])
    #表5：评论分数数据表（jdata_user_comment_score）
    #user_comment_score = pd.read_csv(path+"jdata_user_comment_score.csv",parse_dates=['comment_create_tm'])

    # merge order
    df_order = pd.merge(user_order,sku_basic_info,how='left',on=['sku_id'])
    df_order = pd.merge(df_order,user_basic_info,how='left',on=['user_id'])
    # create year/month/day feature
    df_order['yaer'] = df_order['o_date'].map(lambda x:x.year)
    df_order['month'] = df_order['o_date'].map(lambda x:x.month)
    df_order['day'] = df_order['o_date'].map(lambda x:x.day)

    # merge action
    df_action = pd.merge(user_action,sku_basic_info,how='left',on=['sku_id'])
    # create year/month/day feature
    df_action['yaer'] = df_action['a_date'].map(lambda x:x.year)
    df_action['month'] = df_action['a_date'].map(lambda x:x.month)
    df_action['day'] = df_action['a_date'].map(lambda x:x.day)
    
    
    return df_order, df_action

def create_feat(df_order,df_action,begin_date,end_date,is_train):
    # tranform date, 必须要保证此日期为当月的最后天
    begin_date = pd.to_datetime(begin_date)
    end_date = pd.to_datetime(end_date)
    # result
    result = pd.DataFrame()
    
    for date in pd.date_range(begin_date,end_date,freq='M'):
        order = create_order_feat(df_order, date, date, is_train)
        action = create_action_feat(df_action, date, date)
        lm_order = create_order_feat_lm(df_order,date+2)
        
        df = pd.merge(order,action,how='left',on=['user_id'])
        df = df.merge(lm_order,how='left',on=['user_id'])
        
        result = pd.concat([result,df])
    
    return result

# find the lastest month info
def create_order_feat_lm(df,date):
    # find this month
    order = df[df.month==date.month]
    # user list
    result = pd.DataFrame(order.user_id.unique(),columns=['user_id'])
    # is_buy
    result['lm_is_buy'] = 1
    # order times
    lm_order_times = order.groupby(['user_id'])['o_id'].nunique().reset_index().\
                        rename(columns={'o_id':'lm_order_times'})
    result = result.merge(lm_order_times,how='left',on=['user_id'])
    # sku total num
    lm_sku_total_num = order.groupby(['user_id'])['o_sku_num'].sum().reset_index().\
                        rename(columns={'o_sku_num':'lm_sku_total_num'})
    result = result.merge(lm_sku_total_num,how='left',on=['user_id'])
    # sku_id num
    lm_sku_id_num = order.groupby(['user_id'])['sku_id'].nunique().reset_index().\
                        rename(columns={'sku_id':'lm_sku_id_num'})
    result = result.merge(lm_sku_id_num,how='left',on=['user_id'])
    
    # price mean
    lm_o_mean_price = order.groupby(['user_id'])['price'].mean().reset_index().\
                        rename(columns={'price':'lm_o_mean_price'})
    result =  result.merge(lm_o_mean_price,how='left',on=['user_id'])
    
    # price median
    lm_o_median_price = order.groupby(['user_id'])['price'].median().reset_index().\
                        rename(columns={'price':'lm_o_median_price'})
    result =  result.merge(lm_o_median_price,how='left',on=['user_id'])
    
    # price max 
    lm_o_max_price = order.groupby(['user_id'])['price'].max().reset_index().\
                        rename(columns={'price':'lm_o_max_price'})
    result =  result.merge(lm_o_max_price,how='left',on=['user_id'])
    
    # price min
    lm_o_min_price = order.groupby(['user_id'])['price'].min().reset_index().\
                        rename(columns={'price':'lm_o_min_price'})
    result =  result.merge(lm_o_min_price,how='left',on=['user_id'])
    
    return result


# no label, use "2018-06-30", the monst days
def create_order_feat(df, begin_date, end_date, is_train):
    # tranform date, 必须要保证此日期为当月的最后天
    begin_date = pd.to_datetime(begin_date)
    end_date = pd.to_datetime(end_date)
    # result
    result = pd.DataFrame()
    
    for date in pd.date_range(begin_date,end_date,freq='M'):
        # use 3 months to find feature
        order = df[(df.month==date.month) | (df.month==(date+1).month) | (df.month==(date+2).month)]
        # find feature
        order = _create_order_feat(order, date+3)
        # find label by the 4st month
        if is_train == 1:    
            label = df[df.month==(date+3).month]
            label['buy'] = 1
            label = label[['user_id', 'buy', 'o_date']]
            # sort by o_date
            label = label.sort_values(by=['o_date'])
            # drop_duplicates and remainder the fisrt day
            label = label[label['user_id'].duplicated()==False]
            # find days between the beginning of the month
            label['day'] = label['o_date'].map(lambda x: x.day)
            # merge feature and label by user_id
            order = order.merge(label[['user_id','buy','day']],how='left',on=['user_id'])
        # concat order
        result = pd.concat([result,order])
    
    # return result    
    return result
    
def _create_order_feat(order, predict_month):
    # find user_id, age, sex, user_lv_cd
    result = pd.DataFrame(order[['user_id','age','sex','user_lv_cd']].drop_duplicates())
    
    # find distance_day between last time and the beginning of the predict month
    predict_begin_date = pd.datetime(predict_month.year,predict_month.month,1)
    # sort by o_date descending order
    order = order.sort_values(by=['user_id','o_date'], ascending=False)
    last_order = order[order['user_id'].duplicated()==False][['user_id','o_date']]
    last_order['distance_day'] = last_order['o_date'].map(lambda x: (predict_begin_date-x).days)
    # merge distance_day
    result = result.merge(last_order[['user_id','distance_day']],how='left',on=['user_id'])
    
    # find distance_month between last time and the beginning of the predict month
    last_order['distance_month'] = last_order['o_date'].map(lambda x: predict_begin_date.month-x.month \
              if predict_begin_date.month-x.month>=0 else predict_begin_date.month-x.month+12)
    # merge distance_month
    result = result.merge(last_order[['user_id','distance_month']],how='left',on=['user_id'])
    
    # find in several different months
    different_months = order.groupby(['user_id'])['month'].nunique().reset_index().\
                        rename(columns={'month':'different_months_num'})
    # merge different_months
    result = result.merge(different_months,how='left',on=['user_id'])
    
    # find separate average day,必须要保证predict_begin_date为当月的第一个天，即1号
    first_month_date = predict_begin_date-pd.tseries.offsets.MonthBegin(3)
    order['dst_first_month_days'] = order['o_date'].map(lambda x: (x-first_month_date).days)   
    sep_ave_day = order.groupby(['user_id'])['dst_first_month_days'].mean().reset_index().\
                        rename(columns={'dst_first_month_days':'sep_ave_day'})
    result = result.merge(sep_ave_day,how='left',on=['user_id'])
    
    # find purchase times in three months
    purchase_times = order.groupby(['user_id'])['o_id'].nunique().reset_index().\
                        rename(columns={'o_id':'purchase_times'})
    result = result.merge(purchase_times,how='left',on=['user_id'])
    
    # find total_purchases in three months
    total_purchases = order.groupby(['user_id'])['o_sku_num'].sum().reset_index().\
                        rename(columns={'o_sku_num':'total_purchases'})
    result = result.merge(total_purchases,how='left',on=['user_id'])
    
    # price mean
    o_mean_price = order.groupby(['user_id'])['price'].mean().reset_index().\
                        rename(columns={'price':'o_mean_price'})
    result =  result.merge(o_mean_price,how='left',on=['user_id'])
    
    # price median
    o_median_price = order.groupby(['user_id'])['price'].median().reset_index().\
                        rename(columns={'price':'o_median_price'})
    result =  result.merge(o_median_price,how='left',on=['user_id'])
    
    # price max 
    o_max_price = order.groupby(['user_id'])['price'].max().reset_index().\
                        rename(columns={'price':'o_max_price'})
    result =  result.merge(o_max_price,how='left',on=['user_id'])
    
    # price min
    o_min_price = order.groupby(['user_id'])['price'].min().reset_index().\
                        rename(columns={'price':'o_min_price'})
    result =  result.merge(o_min_price,how='left',on=['user_id'])
    
    # o_area
    order = order.sort_values(by=['user_id','o_area'], ascending=False)
    o_area = order[order['user_id'].duplicated()==False][['user_id','o_area']]
    result = result.merge(o_area,how='left',on=['user_id'])
    
    return result
    

def create_action_feat(df, begin_date, end_date):
    # trainform date,必须要保证此日期为当月的最后天
    begin_date = pd.to_datetime(begin_date)
    end_date = pd.to_datetime(end_date)
    
    # all_result
    all_result = pd.DataFrame()
    
    for date in pd.date_range(begin_date,end_date,freq='M'):
        # use 3 months create feature
        action = df[(df.month==date.month) | (df.month==(date+1).month) | (df.month==(date+2).month)]
        
        # find user_id
        result = pd.DataFrame(action['user_id'].unique(),columns=['user_id'])
        
        # look days
        action_look = action[action.a_type==1]
        look_days = action_look.groupby(['user_id'])['a_date'].nunique().reset_index().\
                        rename(columns={'a_date':'look_days'})
        result = result.merge(look_days,how='left',on=['user_id'])
        
        look_total_times = action_look.groupby(['user_id'])['a_num'].sum().reset_index().\
                        rename(columns={'a_num':'look_total_times'})
        result = result.merge(look_total_times,how='left',on=['user_id'])
        
        # look separate days
        begin_date_1 = pd.datetime(begin_date.year,begin_date.month,1)
        action_look['dst_first_month_look_days'] = action_look['a_date'].map(lambda x:(x-begin_date_1).days)
        seq_look_ave_day = action_look.groupby(['user_id'])['dst_first_month_look_days'].mean().reset_index().\
                        rename(columns={'dst_first_month_look_days':'seq_look_ave_day'})
        result = result.merge(seq_look_ave_day,how='left',on=['user_id'])
        
        # last look days between last look date and predict date
        predict_date_1 = begin_date_1+pd.tseries.offsets.MonthBegin(3)
        # sort by a_date descending order
        action_look = action_look.sort_values(by=['user_id','a_date'], ascending=False)
        last_look = action_look[action_look['user_id'].duplicated()==False][['user_id','a_date']]
        last_look['last_look_days'] = last_look['a_date'].map(lambda x: (predict_date_1-x).days)
        result = result.merge(last_look[['user_id','last_look_days']],how='left',on=['user_id'])
        
        # is_attention
        action_attention = action[action.a_type==2]
        action_attention_unique = pd.DataFrame(action_attention['user_id'].unique(),columns=['user_id'])
        action_attention_unique['is_attention'] = 1
        result = result.merge(action_attention_unique,how='left',on=['user_id'])
        
        # concat
        all_result = pd.concat([all_result,result])
    
    # return 
    return all_result
        
# offline score
def score(pred, real):
    # pred: user_id, pre_date | real: user_id, o_date
    # wi与oi的定义与官网相同
    pred['pred_day'] = pd.to_datetime(pred['pred_date']).dt.day
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    real['real_day'] = pd.to_datetime(real['o_date']).dt.day
    real['oi'] = 1

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare.fillna(0, inplace=True)  # 实际上没有购买的用户，correct_for_S1列的值为nan，将其赋为0
    S1 = np.sum(compare['oi'] * compare['wi']) / np.sum(compare['wi'])

    compare_for_S2 = compare[compare['oi'] == 1]
    S2 = np.sum(10 / (10 + np.square(compare_for_S2['pred_day'] - compare_for_S2['real_day']))) / real.shape[0]

    S = 0.4 * S1 + 0.6 * S2
    print("S1=", S1, "| S2 ", S2)
    print("S =", S)
    
def find_fist_buy_day(real):
    result = pd.DataFrame()
    # sort by o_date
    real = real.sort_values(by=['user_id','o_date'])
    # find user first time
    result = real[real['user_id'].duplicated()==False][['user_id','o_date']]        
    return result
    