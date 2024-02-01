def init_dataset_setting_commercial():

    global load_path, ckpt, user_vocab, item_vocab, src_session_vocab, train_file, valid_file, test_file,\
        JSR_train_file, item_id_num, item_id_dim, item_type1_num, item_type1_dim, item_cate_num, item_cate_dim,\
            user_id_num, user_id_dim, user_gender_num, user_gender_dim, user_age_num, user_age_dim, user_src_level_num, user_src_level_dim,\
                query_id_num, query_id_dim, query_search_source_num, query_search_source_dim, query_word_segs_num, query_word_segs_dim,\
                    max_rec_his_len, max_words_of_item, max_words_of_query, max_src_his_len, max_src_click_item, item2query_vocab,\
                        item_province_name_num,item_province_name_dim,\
                            user_gender_num,user_gender_dim,user_age_range_num,user_age_range_dim,user_fre_city_level_num,user_fre_city_level_dim,\
                                user_fre_country_region_num,user_fre_country_region_dim,user_user_active_degree_num,user_user_active_degree_dim,\
                                    max_satis_his_len,max_dissatis_his_len,item_first_level_category_id_num,item_first_level_category_id_dim,item_second_level_category_id_num,item_second_level_category_id_dim

    """data files info"""

    load_path = './data/commercial'
    ckpt = 'ckpt'

    user_vocab = 'vocab/commercial_user_vocab.pickle'
    item_vocab = 'vocab/commercial_item_vocab.pickle'

    train_file = 'dataset/commercial_train.tsv'
    valid_file = 'dataset/commercial_valid.tsv'
    test_file = 'dataset/commercial_test.tsv'
    

    """item/user/query feature"""

    item_id_num = 2150000 #zero for padding
    item_id_dim = 32 
    item_province_name_num = 600  
    item_province_name_dim = 8
    item_first_level_category_id_num=40
    item_first_level_category_id_dim=8
    item_second_level_category_id_num=175
    item_second_level_category_id_dim=16
 
    user_id_num = 10960
    user_id_dim = 16
    user_gender_num = 3
    user_gender_dim = 4
    user_age_range_num = 8
    user_age_range_dim = 4
    user_fre_city_level_num=8
    user_fre_city_level_dim=4
    user_fre_country_region_num=3
    user_fre_country_region_dim=4
    user_user_active_degree_num=9
    user_user_active_degree_dim=4


    """experiment config"""
    max_rec_his_len = 100
    max_satis_his_len = 25
    max_dissatis_his_len = 5

def init_dataset_setting_kuairand():

    global load_path, ckpt, user_vocab, item_vocab, src_session_vocab, train_file, valid_file, test_file,\
        JSR_train_file, item_id_num, item_id_dim, item_type1_num, item_type1_dim, item_cate_num, item_cate_dim,\
            user_id_num, user_id_dim, user_gender_num, user_gender_dim, user_age_num, user_age_dim, user_src_level_num, user_src_level_dim,\
                query_id_num, query_id_dim, query_search_source_num, query_search_source_dim, query_word_segs_num, query_word_segs_dim,\
                    max_rec_his_len, max_words_of_item, max_words_of_query, max_src_his_len, max_src_click_item, item2query_vocab,\
                        item_tag_id_num,item_tag_id_dim,item_upload_type_num,item_upload_type_dim,item_province_name_num,item_province_name_dim,\
                            user_gender_num,user_gender_dim,user_age_range_num,user_age_range_dim,user_fre_city_level_num,user_fre_city_level_dim,\
                                user_fre_country_region_num,user_fre_country_region_dim,user_user_active_degree_num,user_user_active_degree_dim,\
                                    max_satis_his_len,max_dissatis_his_len,item_first_level_category_id_num,item_first_level_category_id_dim,item_second_level_category_id_num,item_second_level_category_id_dim,\
                        item_author_id_num,item_author_id_dim,item_like_cnt_num,item_like_cnt_dim,user_user_gender_num,user_user_gender_dim,user_user_level_num,user_user_level_dim

    """data files info"""

    # load_path = './data/8k_test'
    load_path='./data_all/data_week_1w'
    # load_path='../SZH_two_task/data_all/data_week_1w'
    # load_path='../mtl/data_all/data_week_1w'
    ckpt = 'ckpt'

    # user_vocab = 'vocab/user_vocab.pickle'
    # item_vocab = 'vocab/8k_item.pickle'
    # user_vocab = 'vocab/1019_week_user_info_his3.pickle'
    # user_vocab = 'vocab/1019_week_user_info.pickle'
    # item_vocab = 'vocab/item_info_0911.pickle'
    # item_vocab = 'vocab/1019_week_item_info_tag.pickle'
    # user_vocab = 'vocab/1019_week_user_info_his3.pickle'
    user_vocab = 'vocab/1219_week_user_info_his3_no_time.pickle'
    item_vocab = 'vocab/1114_week_item_info.pickle'

    #train_file = 'dataset/train_inter.tsv'
    #valid_file = 'dataset/valid_inter.tsv'
    #test_file = 'dataset/test_inter.tsv'
    # train_file = 'dataset/re_filter_train_min.tsv'
    # valid_file = 'dataset/re_filter_val_min.tsv'
    # test_file = 'dataset/re_filter_test_min.tsv'
    # train_file = 'dataset/rere_filter_train.tsv'
    # valid_file = 'dataset/rere_filter_val.tsv'
    # test_file = 'dataset/rere_filter_test.tsv'
    # train_file = 'dataset/re_re_sel_data_train_min.tsv'
    # valid_file = 'dataset/re_re_sel_data_val_min.tsv'
    # test_file = 'dataset/re_re_sel_data_test_min.tsv'
    # train_file = 'dataset/re_re_sel_data_train.tsv'
    # valid_file = 'dataset/re_re_sel_data_val.tsv'
    # test_file = 'dataset/re_re_sel_data_test.tsv'
    
    # train_file = 'dataset/1014_sel_data_train_min.tsv'
    # valid_file = 'dataset/1014_sel_data_val_min.tsv'
    # test_file = 'dataset/1014_sel_data_test_min.tsv'
    # train_file = 'dataset/1014_sel_data_train.tsv'
    # valid_file = 'dataset/1014_sel_data_val.tsv'
    # test_file = 'dataset/1014_sel_data_test.tsv'
    # train_file = 'dataset/1014_ques_train.tsv'
    # valid_file = 'dataset/1014_ques_val.tsv'
    # test_file = 'dataset/1014_ques_test.tsv'


    # train_file = 'dataset/1225_sel_data_train.tsv'
    # valid_file = 'dataset/1225_sel_data_val.tsv'
    # test_file = 'dataset/1225_sel_data_test.tsv'
    # test_file = 'dataset/19_test_4_4.tsv'
    
    # test_file = 'dataset/111_pos_ques_val.tsv'
    
    train_file = 'dataset/1225_sel_data_train_min.tsv'
    valid_file = 'dataset/1225_sel_data_val_min.tsv'
    test_file = 'dataset/1225_sel_data_test_min.tsv'

    # train_file = 'dataset/re_1019_sel_data_train_min.tsv'
    # valid_file = 'dataset/re_1019_sel_data_val_min.tsv'
    # test_file = 'dataset/re_1019_sel_data_test_min.tsv'
    # train_file = 'dataset/re_re_1019_sel_data_train.tsv'
    # valid_file = 'dataset/re_re_1019_sel_data_val.tsv'
    # test_file = 'dataset/re_re_1019_sel_data_test.tsv'
    # train_file = 'dataset/re_1019_ques_train.tsv'
    # valid_file = 'dataset/re_1019_ques_val.tsv'
    # test_file = 'dataset/re_1019_ques_test.tsv'
    
    # train_file = 'dataset/ques_year_sel_data_train_min.tsv'
    # valid_file = 'dataset/ques_year_sel_data_val_min.tsv'
    # test_file = 'dataset/ques_year_sel_data_test_min.tsv'
    
    # train_file = 'dataset/ques_year_sel_data_train.tsv'
    # valid_file = 'dataset/ques_year_sel_data_val.tsv'
    # test_file = 'dataset/test_ques_year_sel_data_test.tsv'
    
    # 第二次训练
    # train_file = 'dataset/ques_year_sel_data_val.tsv'
    # valid_file = 'dataset/test_ques_year_sel_data_val.tsv'
    # test_file = 'dataset/test_ques_year_sel_data_test.tsv'
    # test_file = 'dataset/1104_sel_data_test.tsv'
    
    # 第三次训练
    # train_file = 'dataset/test_ques_year_sel_data_val.tsv'
    # valid_file = 'dataset/test_ques_year_sel_data_val.tsv'
    # test_file = 'dataset/test_ques_year_sel_data_test.tsv'

    """item/user/query feature"""

    # hidden for privacy item 64+8+4=76 user:64+20=84
    # item_id_num = 9999999 #zero for padding
    item_id_num = 2150000 #zero for padding
    item_id_dim = 32 
    item_province_name_num = 600  #author_id太大了，需要的词典太大
    item_province_name_dim = 8
    # item_upload_type_num = 99  #需要统计最大的like_cnt
    # item_upload_type_dim = 4  
    item_first_level_category_id_num=40
    item_first_level_category_id_dim=8
    item_second_level_category_id_num=175
    item_second_level_category_id_dim=16
 
    user_id_num = 10960
    user_id_dim = 16
    user_gender_num = 3
    user_gender_dim = 4
    user_age_range_num = 8
    user_age_range_dim = 4
    user_fre_city_level_num=8
    user_fre_city_level_dim=4
    user_fre_country_region_num=3
    user_fre_country_region_dim=4
    user_user_active_degree_num=9
    user_user_active_degree_dim=4


    """experiment config"""
    max_rec_his_len = 100
    max_satis_his_len = 25
    max_dissatis_his_len = 5
