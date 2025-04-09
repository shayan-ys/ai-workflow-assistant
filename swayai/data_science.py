import pandas as pd


def infer_column_types(df: pd.DataFrame) -> dict:
    inferred = {}

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(10).tolist()
        types_seen = set()

        for val in sample_values:
            val = val.strip().replace(",", "").replace("$", "")

            if val.lower() in ["true", "false", "yes", "no", "0", "1"]:
                types_seen.add("bool")
            elif val.replace(".", "", 1).isdigit():
                if "." in val:
                    types_seen.add("float")
                else:
                    types_seen.add("int")
            elif any(x in val for x in ["$", "usd"]):
                types_seen.add("currency")
            else:
                types_seen.add("string")

        # Pick dominant type
        if "string" in types_seen:
            inferred[col] = "string"
        elif "currency" in types_seen:
            inferred[col] = "currency"
        elif "float" in types_seen:
            inferred[col] = "float"
        elif "int" in types_seen:
            inferred[col] = "int"
        elif "bool" in types_seen:
            inferred[col] = "bool"
        else:
            inferred[col] = "unknown"

    return inferred
