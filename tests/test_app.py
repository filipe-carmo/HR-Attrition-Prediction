"""Smoke tests for the Gradio app's prediction functions."""

import pytest


def _predict(app, **overrides):
    inputs = dict(
        job_satisfaction=3, job_involvement=3, environment_satisfaction=3, work_life_balance=3,
        num_companies_worked=2, total_working_years=10, years_at_company=5,
        department="Research & Development", job_role="Laboratory Technician",
        business_travel="Travel_Rarely", marital_status="Married", overtime="No",
        model_name=app.LOGREG_LABEL,
    )
    inputs.update(overrides)
    return app.predict_attrition(**inputs)


def test_high_risk_profile_scores_above_low_risk_profile(app_module):
    risky, *_ = _predict(
        app_module, job_satisfaction=1, environment_satisfaction=1, work_life_balance=1,
        num_companies_worked=6, total_working_years=2, years_at_company=1, department="Sales",
        job_role="Sales Representative", business_travel="Travel_Frequently",
        marital_status="Single", overtime="Yes",
    )
    safe, *_ = _predict(app_module, job_satisfaction=4, environment_satisfaction=4, total_working_years=20,
                        years_at_company=15, job_role="Manager")
    assert 0 <= safe < risky <= 1
    assert risky >= app_module.THRESHOLD_LOGREG


@pytest.mark.parametrize("role", ["Sales Executive", "Research Scientist", "Human Resources"])
def test_baseline_job_roles_are_accepted(app_module, role):
    proba, label, risk, _ = _predict(app_module, job_role=role, model_name=app_module.DT_LABEL)
    assert label != "❌ Error"
    assert 0 <= proba <= 1


def test_batch_prediction_accepts_a_file_path(app_module, raw_df, tmp_path):
    csv = tmp_path / "employees.csv"
    raw_df.head(25).to_csv(csv, index=False)
    summary, results = app_module.predict_batch(str(csv), app_module.LOGREG_LABEL)
    assert "Total Employees:** 25" in summary
    assert results["Risk_Level"].notna().all()
    assert set(results["Prediction_Label"]) <= {"Stay", "Leave"}


def test_batch_prediction_reports_missing_columns(app_module, raw_df, tmp_path):
    csv = tmp_path / "employees.csv"
    raw_df.drop(columns=["OverTime"]).head(5).to_csv(csv, index=False)
    summary, results = app_module.predict_batch(str(csv), app_module.LOGREG_LABEL)
    assert "OverTime" in summary
    assert results is None
