# first line: 910
@_memory.cache
def load_results(include_pruning=False, include_goog_results=False,
                 include_lottery_ticket=False, include_efficientnet=False,
                 clean=True, **clean_kwargs):
    df = None
    if include_pruning:
        df = load_pruning_results()
    if include_goog_results:
        df_goog = load_state_of_sparsity_resnet_results()
        df = df_goog if df is None else pd.concat(
            (df, df_goog), axis=0, ignore_index=True, sort=False)
    if include_lottery_ticket:
        df_lot = load_lottery_ticket_resnet_results()
        df = df_lot if df is None else pd.concat(
            (df, df_lot), axis=0, ignore_index=True, sort=False)
    if include_efficientnet:
        df_eff = load_efficientnet_results()
        df = df_eff if df is None else pd.concat(
            (df, df_eff), axis=0, ignore_index=True, sort=False)

    if clean:
        df = clean_results(df, **clean_kwargs)

    return df
