"""
Mediation Model 3: Full outcome (direct effect + path b).
Y = EMP_Mean or EMP_1 ~ condition_avatar + anthropomorphism_score + controls.
Output: model3_full_outcome/EMP_Mean/, model3_full_outcome/EMP_1/
"""
import os
import pandas as pd
import statsmodels.formula.api as smf

from viz_helpers import save_mediation_figures


def count_tokens(text: str) -> int:
    if pd.isna(text):
        return 0
    return len(str(text).split())


def add_personality_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["extraversion_score"] = (
        df["P3_do_not_mind_centre_of_attention"]
        + df["P10_make_friends_easily"]
        - df["P14_keep_in_the_background"]
        - df["P18_avoid_contact_with_others"]
    ) / 4
    df["agreeableness_score"] = (
        -df["P2_hold_a_grudge"]
        + df["P6_believe_others_have_good_intentions"]
        - df["P9_cut_others_to_pieces"]
        + df["P13_am_easy_to_satisfy"]
    ) / 4
    return df


def add_anthropomorphism_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    anthro_cols = ["G_ANTHRO_1", "G_ANTHRO_2", "G_ANTHRO_3", "G_ANTHRO_4", "G_ANTHRO_5"]
    df["anthropomorphism_score"] = df[anthro_cols].mean(axis=1)
    return df


def build_and_save(df: pd.DataFrame, outcome: str, formula: str, output_dir: str) -> None:
    model = smf.ols(formula=formula, data=df).fit()
    outcome_dir = os.path.join(output_dir, outcome)
    os.makedirs(outcome_dir, exist_ok=True)

    y_true = df[outcome]
    y_pred = model.predict(df)
    pred_df = pd.DataFrame({
        "person_id": df["person_id"].values,
        "actual": y_true.values,
        "predicted": y_pred.values,
        "residual": (y_true - y_pred).values,
        "condition": df["condition"].values,
        "condition_avatar": df["condition_avatar"].values,
        "anthropomorphism_score": df["anthropomorphism_score"].values,
        "total_tokens": df["total_tokens"].values,
        "Sex": df["Sex"].values,
        "Ethnicity simplified": df["Ethnicity simplified"].values,
        "extraversion_score": df["extraversion_score"].values,
        "agreeableness_score": df["agreeableness_score"].values,
    })
    round_cols = ["actual", "predicted", "residual", "anthropomorphism_score", "total_tokens", "extraversion_score", "agreeableness_score"]
    for col in round_cols:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").round(4)
    pred_df.to_csv(os.path.join(outcome_dir, "predictions.csv"), index=False)

    save_mediation_figures(df, outcome, y_true, y_pred, outcome_dir)

    coeffs = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    if "P>|t|" in coeffs.columns:
        coeffs["Significance"] = coeffs["P>|t|"].apply(
            lambda p: "*" if p < 0.05 else ("**" if p < 0.1 else "ns")
        )
    coeffs = coeffs.rename(columns={"Coef.": "coefficients", "Std.Err.": "std", "P>|t|": "p-value"})
    coeffs = coeffs[["term", "coefficients", "std", "t", "p-value", "Significance"]]
    for col in coeffs.columns:
        if col != "term" and pd.api.types.is_numeric_dtype(coeffs[col]):
            coeffs[col] = pd.to_numeric(coeffs[col], errors="ignore").round(4)
    coeffs.to_csv(os.path.join(outcome_dir, "coeffs.csv"), index=False)

    fit_stats = pd.DataFrame({
        "outcome": [outcome],
        "n": [int(model.nobs)],
        "r_squared": [model.rsquared],
        "adj_r_squared": [model.rsquared_adj],
        "aic": [model.aic],
        "bic": [model.bic],
    })
    fit_stats.to_csv(os.path.join(outcome_dir, "fit_stats.csv"), index=False)
    print(f"  {outcome}: coeffs, fit_stats, predictions -> {outcome_dir}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    survey_path = os.path.join(base_dir, "..", "survey_results", "survey_responses.csv")
    conversation_path = os.path.join(base_dir, "..", "survey_results", "conversation.csv")

    df_survey = pd.read_csv(survey_path)
    df_survey["EMP_Mean"] = df_survey[["EMP_1", "EMP_2", "EMP_3", "EMP_4", "EMP_5"]].mean(axis=1)
    df_survey = add_personality_scores(df_survey)
    df_survey = add_anthropomorphism_score(df_survey)

    df_conversation = pd.read_csv(conversation_path)
    df_conversation["token_count"] = df_conversation["text"].apply(count_tokens)
    df_tokens = (
        df_conversation.groupby("person_id")["token_count"]
        .sum()
        .reset_index()
        .rename(columns={"token_count": "total_tokens"})
    )
    df = pd.merge(df_survey, df_tokens, on="person_id", how="left")
    df = df[df["condition"].isin(["male_avatar", "female_avatar", "text"])].copy()
    df["condition_avatar"] = df["condition"].isin(["male_avatar", "female_avatar"]).astype(int)

    required_columns = [
        "EMP_Mean", "EMP_1", "condition_avatar", "anthropomorphism_score",
        "Sex", "Ethnicity simplified", "extraversion_score", "agreeableness_score", "total_tokens"
    ]
    df_model = df.dropna(subset=required_columns).copy()
    print(f"Model 3 (Full outcome: direct effect + path b). Rows: {len(df_model)}")

    output_dir = os.path.join(base_dir, "model3_full_outcome")
    formula_base = "{} ~ condition_avatar + anthropomorphism_score + extraversion_score + agreeableness_score + C(Sex) + C(Q('Ethnicity simplified')) + total_tokens"
    build_and_save(df_model, "EMP_Mean", formula_base.format("EMP_Mean"), output_dir)
    build_and_save(df_model, "EMP_1", formula_base.format("EMP_1"), output_dir)


if __name__ == "__main__":
    main()
