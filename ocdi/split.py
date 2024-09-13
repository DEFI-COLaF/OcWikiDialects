"""Module with function to split a dataset into train/dev/test splits"""
import pandas as pd


def split_train_dev_test(
        df: pd.DataFrame,
        col_dialect="dialect",
        col_text="text",
        test_strategy="balance",
        dev_strategy="balance",
        test_size=None,
        dev_size=None,
        max_ratio=0.5,
        min_threshold=10,
        max_hard=500,
        seed=1,
        unit_multiplier_col=None
):

    def get_max_per_dialect(df, max_ratio=0.5, min_threshold=10, max_hard=500, unit_multiplier_col=None):

        def get_max(count, max_ratio, threshold, max_hard):
            """Use max 50% of the data, or all the data if <=10"""
            if count <= threshold:
                return count
            # Restrain the count to the max_ratio (e.g. 50% of samples)
            count = round(count * max_ratio)
            # Restrain the count to a hard maximum value (e.g. max 500 samples)
            if max_hard:
                count = min(count, max_hard)
            return count

        if unit_multiplier_col:
            # Get counts by unit
            df_dia_count = df.groupby(col_dialect)[unit_multiplier_col].sum().to_frame("count")
        else:
            df_dia_count = df.value_counts(col_dialect).to_frame("count")
        df_dia_count["max_split"] = df_dia_count["count"].apply(
            get_max, max_ratio=max_ratio, threshold=min_threshold, max_hard=max_hard)
        df_dia_count.sort_index(inplace=True)
        max_per_dialect = df_dia_count["max_split"].to_dict()
        return max_per_dialect

    def set_dia_freqs(df):
        dia_freqs = (df[col_dialect].value_counts() / len(df)).to_dict()
        df[COL_DIA_WEIGHTS] = df.apply(lambda row: dia_freqs[row[col_dialect]], axis=1)

    def select_balanced_samples(df, dialects, n=1000, max_per_dialect=None, unit_multiplier_col=None):
        ids_selection = []
        total_selected = 0
        dia_dfs = {d: df[df[col_dialect] == d] for d in dialects}
        sel_per_dialect_samples = {d: 0 for d in dialects}  # Takes into account possible future expansion into units
        sel_per_dialect_ids = {d: 0 for d in dialects}
        max_reached = set()

        # Check that there are enough samples to pick from
        if max_per_dialect:
            sum_avail = sum(max_per_dialect.values())
            assert sum_avail >= n, \
                (f"Not enough samples to pick from given the max_per_dialect "
                 f"({n} < {sum_avail} --> {max_per_dialect})")

        while total_selected < n:
            for dialect in dialects:
                # Check if max is reached
                max_samples = max_per_dialect[dialect] if max_per_dialect else None
                if total_selected >= n:
                    break
                if max_samples is not None and sel_per_dialect_samples[dialect] >= max_samples:
                    max_reached.add(dialect)
                    continue
                # Pick sample
                next_id = sel_per_dialect_ids[dialect]
                sample = dia_dfs[dialect].iloc[next_id]
                # Update counters
                if unit_multiplier_col:
                    # For OcWikiDialects, select article IDs,
                    # but count the number of paragraphs that will be expanded later
                    n_sample_units = sample[unit_multiplier_col]
                    if n_sample_units == 0:
                        continue
                    total_selected += n_sample_units
                    sel_per_dialect_samples[dialect] += n_sample_units
                    sel_per_dialect_ids[dialect] += 1
                else:
                    total_selected += 1
                    sel_per_dialect_samples[dialect] += 1
                    sel_per_dialect_ids[dialect] += 1
                # Store the selection
                sample_id = sample.name
                ids_selection.append(sample_id)
            if len(max_reached) == len(dia_dfs):
                raise ValueError(f"Unable to select {n} test samples")

        df_selection = df.loc[ids_selection]
        return df_selection

    def select_stratified_samples(df, n=1000, seed=1):
        df_selection = df.sample(n, weights=COL_DIA_WEIGHTS, random_state=seed)
        return df_selection

    df = df.copy()
    COL_DIA_WEIGHTS = "dia_freq"

    # Remove duplicates - within dialect subsets
    df.drop_duplicates(subset=[col_text, col_dialect], keep="first", inplace=True)
    # print(f"{len(df)} samples without duplicates inside dialect subsets")
    df.drop_duplicates(subset=[col_text], keep=False, inplace=True)
    # print(f"{len(df)} samples without multi-label samples")

    # Get list of dialects and some stats
    max_per_dialect_dict = get_max_per_dialect(df, max_ratio, min_threshold, max_hard, unit_multiplier_col)
    dialects = list(max_per_dialect_dict)
    set_dia_freqs(df)

    # Get test and val sizes
    if test_size is None:
        test_size = sum(max_per_dialect_dict.values())
    elif test_size <= 1.0:
        test_size = round(len(df) * test_size)
    if dev_size is not None and dev_size <= 1.0:
        dev_size = round(len(df) * dev_size)

    # Select test samples
    if test_size > 0:
        if test_strategy == "balance":
            df_test = select_balanced_samples(df, dialects, test_size, max_per_dialect_dict, unit_multiplier_col)
        elif test_strategy == "stratify":
            df_test = select_stratified_samples(df, test_size, seed)
        else:
            raise ValueError(f"Unrecognized sampling strategy: {test_strategy}")
        df.drop(df_test.index, inplace=True)
    else:
        df_test = None

    # Select dev samples
    if dev_size is None or dev_size > 0:
        if dev_strategy == "balance":
            max_per_dialect_dict = get_max_per_dialect(df, max_ratio, min_threshold, max_hard, unit_multiplier_col)
            if dev_size is None:
                dev_size = sum(max_per_dialect_dict.values())
            df_dev = select_balanced_samples(df, dialects, dev_size, max_per_dialect_dict, unit_multiplier_col)
        elif dev_strategy == "stratify":
            assert dev_size is not None, f"dev_size must be set for the stratify strategy"
            df_dev = select_stratified_samples(df, dev_size, seed)
        else:
            raise ValueError(f"Unrecognized sampling strategy: {dev_strategy}")
        df.drop(df_dev.index, inplace=True)
    else:
        df_dev = None

    # Use the rest as training split
    df_train = df

    # Remove "dia_freq" column
    splits = {"train": df_train, "dev": df_dev, "test": df_test}
    for df_split in splits.values():
        if df_split is None:
            continue
        df_split.drop(columns=COL_DIA_WEIGHTS, inplace=True)

    return splits


def load_csv(path):
    """Function to load a LIDoc split CSV file"""
    df = pd.read_csv(
        path,
        usecols=["text", "dialect"],
        header=0,
        na_filter=False
    )
    return df
