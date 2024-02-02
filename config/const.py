def init_dataset_setting_commercial():

    global load_path, ckpt, user_vocab, item_vocab, train_file, valid_file, test_file,\
        item_id_num, item_id_dim,user_id_num, user_id_dim, user_gender_num, user_gender_dim, \
            max_rec_his_len,item_province_name_num,item_province_name_dim,user_gender_num,user_gender_dim,\
                user_age_range_num,user_age_range_dim,user_fre_city_level_num,user_fre_city_level_dim,user_fre_country_region_num,\
                    user_fre_country_region_dim,user_user_active_degree_num,user_user_active_degree_dim,max_satis_his_len,max_dissatis_his_len,\
                        item_first_level_category_id_num,item_first_level_category_id_dim,item_second_level_category_id_num,item_second_level_category_id_dim

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

    global load_path, ckpt, user_vocab, item_vocab, train_file, valid_file, test_file,\
         item_id_num, item_id_dim,user_id_num, user_id_dim, user_gender_num, user_gender_dim,\
            user_onehot_feat2_num,user_onehot_feat2_dim,user_onehot_feat5_num,user_onehot_feat5_dim,\
                user_onehot_feat7_num,user_onehot_feat7_dim,user_onehot_feat8_num,user_onehot_feat8_dim,\
                    max_rec_his_len,item_upload_type_num,item_upload_type_dim,item_province_name_num,item_province_name_dim,\
                        user_gender_num,user_gender_dim,user_age_range_num,user_age_range_dim,user_fre_city_level_num,user_fre_city_level_dim,\
                            user_fre_country_region_num,user_fre_country_region_dim,user_user_active_degree_num,user_user_active_degree_dim,item_music_type_num,item_music_type_dim,\
                                max_satis_his_len,max_dissatis_his_len,item_first_level_category_id_num,item_first_level_category_id_dim,item_second_level_category_id_num,item_second_level_category_id_dim\

    """data files info"""

    load_path = './data/kuairand'
    ckpt = 'ckpt'

    user_vocab = 'vocab/kuairand_user_vocab.pickle'
    item_vocab = 'vocab/kuairand_item_vocab.pickle'

    train_file = 'dataset/kuairand_train.tsv'
    valid_file = 'dataset/kuairand_valid.tsv'
    test_file = 'dataset/kuairand_test.tsv'

    """item/user/query feature"""
    item_id_num = 142257 #zero for padding
    item_id_dim = 32 
    item_music_type_num = 9  
    item_music_type_dim = 4
    item_upload_type_num=29
    item_upload_type_dim=8
 
    user_id_num = 7272
    user_id_dim = 16
    user_onehot_feat2_num = 52
    user_onehot_feat2_dim = 4
    user_onehot_feat5_num = 36
    user_onehot_feat5_dim = 4
    user_onehot_feat7_num = 120
    user_onehot_feat7_dim = 8
    user_onehot_feat8_num = 455
    user_onehot_feat8_dim = 8


    """experiment config"""
    max_rec_his_len = 50
    max_satis_his_len = 20
    max_dissatis_his_len = 5
