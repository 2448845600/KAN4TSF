exp_conf = dict(
    model_name="iKransformer",
    dataset_name='ETTh1',

    hist_len=96,
    pred_len=96,

    output_attention=False,
    d_model=512,
    d_ff=512,
    dropout=0.1,
    factor=3,
    n_heads=8,
    activation='gelu',
    e_layers=2,

    kan_type="WavKAN",
    kan_param=5,
    use_out_kan=False,
    
    lr=0.001,
)
