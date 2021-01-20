model_config = ['create_model_vanilla',
                'create_model_vanilla_dropout_batchnormalization',
                'create_model_stacked',
                'create_model_stacked_dropout_batchnormalization',
                'create_model_1d_cnn']

data_config = ['load_data_beijing_multi',
                'load_data_appliances_multi',
                'load_data_solar_multi',
                'load_data_load_consumption_multi']

input_output_size = \
[
    ### load_data_beijing_multi ###
    {
        "win_size": 24,
        "out_size": 24
    },
    ### load_data_appliances_multi ###
    {
        "win_size": 144,
        "out_size": 144
    },
    ### load_data_solar_multi ###
    {
        "win_size": 24,
        "out_size": 24
    },
    ### load_data_load_consumption_multi ###
    {
        "win_size": 24,
        "out_size": 24
    }
]