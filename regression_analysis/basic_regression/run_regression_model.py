import os
import math
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


MODEL_SPEC = 4
# Control variable specs (for paper):
# 1) condition only
# 2) + demographics (Sex, Ethnicity)
# 3) + demographics + personality (no tokens)
# 4) + demographics + personality + total_tokens


def count_tokens(text: str) -> int:
    if pd.isna(text):
        return 0
    return len(str(text).split())

def add_personality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extraversion and agreeableness as continuous scores (no median-split categories).
    - extraversion_score = (P3 + P10 - P14 - P18) / 4
    - agreeableness_score = (-P2 + P6 - P9 + P13) / 4
    """
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


def build_model(df: pd.DataFrame, outcome: str, base_output_dir: str) -> None:
    # NOTE: We keep all alternative formulas as comments (do not delete) so you can
    # quickly switch MODEL_SPEC between 1-4.
    if MODEL_SPEC == 1:
        formula = f"{outcome} ~ condition_avatar"
    elif MODEL_SPEC == 2:
        formula = f"{outcome} ~ condition_avatar + C(Sex) + C(Q('Ethnicity simplified'))"
    elif MODEL_SPEC == 3:
        formula = f"{outcome} ~ condition_avatar + extraversion_score + agreeableness_score + C(Sex) + C(Q('Ethnicity simplified'))"
    elif MODEL_SPEC == 4:
        formula = f"{outcome} ~ condition_avatar + extraversion_score + agreeableness_score + C(Sex) + C(Q('Ethnicity simplified')) + total_tokens"
    else:
        raise ValueError(f"Unsupported MODEL_SPEC={MODEL_SPEC}. Use 1, 2, 3, or 4.")

    model = smf.ols(formula=formula, data=df).fit()

    outcome_dir = os.path.join(base_output_dir, outcome)
    os.makedirs(outcome_dir, exist_ok=True)

    # Per-person predictions (to help interpret R^2)
    y_true = df[outcome]
    y_pred = model.predict(df)
    pred_df = pd.DataFrame(
        {
            "person_id": df["person_id"].values,
            "actual": y_true.values,
            "predicted": y_pred.values,
            "residual": (y_true - y_pred).values,
            "condition": df["condition"].values,
            "condition_avatar": df["condition_avatar"].values,
            "total_tokens": df["total_tokens"].values,
            "Sex": df["Sex"].values,
            "Ethnicity simplified": df["Ethnicity simplified"].values,
            "extraversion_score": df["extraversion_score"].values,
            "agreeableness_score": df["agreeableness_score"].values,
        }
    )
    for col in ["actual", "predicted", "residual", "total_tokens", "extraversion_score", "agreeableness_score"]:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").round(4)
    predictions_output = os.path.join(outcome_dir, "predictions.csv")
    pred_df.to_csv(predictions_output, index=False)

    # --- Visualization: bar plots by condition and regression line plots ---
    # All plots are saved directly under the regression folder for easy access.
    # 1) Bar plot: mean outcome by experimental condition (text / male_avatar / female_avatar)
    try:
        cond_group = (
            df.groupby("condition")[outcome]
            .agg(["mean", "std", "count"])
            .dropna()
        )
        if not cond_group.empty:
            cond_group["sem"] = cond_group["std"] / cond_group["count"].apply(
                lambda n: math.sqrt(n) if n > 0 else float("nan")
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                cond_group.index,
                cond_group["mean"],
                yerr=cond_group["sem"],
                capsize=4,
                color="#4C72B0",
            )
            ax.set_ylabel(outcome)
            ax.set_xlabel("Condition")
            ax.set_title(f"Mean {outcome} by condition")
            # Perceived empathy items use 1–7 Likert; show full 0–7 scale for comparability
            ax.set_ylim(0, 7)
            ax.set_yticks(list(range(0, 8)))
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            bar_path = os.path.join(
                base_output_dir, f"{outcome}_bar_by_condition.png"
            )
            plt.savefig(bar_path, dpi=300)
            plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to create bar plot for {outcome}: {e}")

    # 2) Regression line-style plot: predicted vs. actual with 45-degree line
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{outcome}: predicted vs. actual")
        plt.tight_layout()
        regline_path = os.path.join(
            base_output_dir, f"{outcome}_pred_vs_actual.png"
        )
        plt.savefig(regline_path, dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to create regression-line plot for {outcome}: {e}")

    coeffs = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    if "P>|t|" in coeffs.columns:
        coeffs["Significance"] = coeffs["P>|t|"].apply(
            lambda p: "*" if p < 0.05 else ("**" if p < 0.1 else "ns")
        )
    coeffs = coeffs.rename(
        columns={"Coef.": "coefficients", "Std.Err.": "std", "P>|t|": "p-value"}
    )
    keep_columns = ["term", "coefficients", "std", "t", "p-value", "Significance"]
    coeffs = coeffs[keep_columns]
    for col in coeffs.columns:
        if col == "term":
            continue
        coeffs[col] = pd.to_numeric(coeffs[col], errors="ignore")
        if pd.api.types.is_numeric_dtype(coeffs[col]):
            coeffs[col] = coeffs[col].round(4)
    coeffs_output = os.path.join(outcome_dir, "coeffs.csv")
    coeffs.to_csv(coeffs_output, index=False)

    fit_stats = pd.DataFrame(
        {
            "outcome": [outcome],
            "n": [int(model.nobs)],
            "r_squared": [model.rsquared],
            "adj_r_squared": [model.rsquared_adj],
            "aic": [model.aic],
            "bic": [model.bic],
        }
    )
    stats_output = os.path.join(outcome_dir, "fit_stats.csv")
    fit_stats.to_csv(stats_output, index=False)

    print(f"{outcome} model saved:")
    print(f"- Coeffs: {coeffs_output}")
    print(f"- Fit stats: {stats_output}")
    print(f"- Predictions: {predictions_output}")


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = base_dir

    survey_path = os.path.join(base_dir, "..", "survey_results", "survey_responses.csv")
    conversation_path = os.path.join(base_dir, "..", "survey_results", "conversation.csv")

    df_survey = pd.read_csv(survey_path)
    emp_cols = ["EMP_1", "EMP_2", "EMP_3", "EMP_4", "EMP_5"]
    df_survey["EMP_Mean"] = df_survey[emp_cols].mean(axis=1)
    # 第一种情况不需要 personality，跑 spec 3–4 时取消下面注释
    df_survey = add_personality_scores(df_survey)

    # total_tokens = 参与者 + 对应 chatbot 的整段对话（按 person_id 汇总）
    df_conversation = pd.read_csv(conversation_path)
    df_conversation["token_count"] = df_conversation["text"].apply(count_tokens)
    df_tokens = (
        df_conversation.groupby("person_id")["token_count"]
        .sum()
        .reset_index()
        .rename(columns={"token_count": "total_tokens"})
    )
    df = pd.merge(df_survey, df_tokens, on="person_id", how="left")

    # Only keep the core experimental conditions to avoid mixing other groups into text/avatar.
    df = df[df["condition"].isin(["male_avatar", "female_avatar", "text"])].copy()
    df["condition_avatar"] = df["condition"].isin(["male_avatar", "female_avatar"]).astype(int)

    # Required columns depend on MODEL_SPEC. We keep all alternatives as comments.
    if MODEL_SPEC == 1:
        required_columns = ["EMP_Mean", "EMP_1", "condition_avatar"]
    elif MODEL_SPEC == 2:
        required_columns = ["EMP_Mean", "EMP_1", "condition_avatar", "Sex", "Ethnicity simplified"]
    elif MODEL_SPEC == 3:
        required_columns = ["EMP_Mean", "EMP_1", "condition_avatar", "Sex", "Ethnicity simplified", "extraversion_score", "agreeableness_score"]
    elif MODEL_SPEC == 4:
        required_columns = ["EMP_Mean", "EMP_1", "condition_avatar", "Sex", "Ethnicity simplified", "extraversion_score", "agreeableness_score", "total_tokens"]
    else:
        raise ValueError(f"Unsupported MODEL_SPEC={MODEL_SPEC}. Use 1, 2, 3, or 4.")

    df_model = df.dropna(subset=required_columns).copy()

    print(f"Rows after filtering: {len(df_model)}")
    build_model(df_model, "EMP_Mean", output_dir)
    build_model(df_model, "EMP_1", output_dir)


if __name__ == "__main__":
    main()
